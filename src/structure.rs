use bevy_rapier3d::prelude::RigidBody;
use std::fs::File;
use std::path::Path;
use bevy::asset::{AssetServer, Handle};
use bevy::pbr::PointLight;
use bevy::prelude::{Color, Commands, Res, Scene, SceneBundle, Transform};
use bevy_prng::WyRand;
use rand::prelude::SliceRandom;
use serde::{Serialize, Deserialize};
use ron::de::{from_reader, SpannedError};

use crate::euler_transform::EulerTransform;
use bevy::prelude::*;
use libnoise::{Generator, Source};
use rand::Rng;
use crate::generation::GenRng;
use crate::material_autoloader::MaterialCache;
use statrs::distribution::{Normal};
use rand::distributions::Distribution;
use bevy_rapier3d::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StructureKey {
    /***** PRIMS ****/
    GLTF {
        //Directly loads a model from the provided path relative to the structures folder
        path: String,
        priority: i8,
        collider: ColliderType,
    },
    #[serde(with = "SerializablePointLight")]
    PointLight ( //Creates a pointlight with the provided data
        PointLight,
    ),
    #[serde(with = "SerializableSpotLight")]
    SpotLight( //Creates a spotlight with the provided data
        SpotLight,
    ),
    SoundEffect(
        String
    ),

    /***** GLOBALS *****/
    #[serde(with = "SerializableDirectionalLight")]
    DirectionalLight( //Overwrite the default directional light
        DirectionalLight
    ),

    #[serde(with = "SerializableAmbientLight")]
    AmbientLight( //Overwrite the default ambient light
        AmbientLight
    ),

    #[serde(with = "SerializableFogSettings")]
    FogSettings( //Overwrite the default fog
        FogSettings
    ),

    BackgroundMusic( //Overwrite the default background music
        String
    ),

    /***** NESTINGS *****/
    Nest ( //Nests a reference, useful for applying custom offsets in things like loops
        StructureReference,
    ),

    /***** MODIFIERS *****/
    Choose { //Choose one of the entries from the provided reference
        list: StructureReference,
    },
    ChooseSome { //Choose some of the entries from the provided reference
        list: StructureReference,
        count: usize,
    },
    Rand { //Jiggle the wrapped reference's transform randomly
        reference: StructureReference,
        rand: RandData,
    },
    ProbabilitySpawn { //Do or don't spawn the referenced structure, based off of a 0-1 probability.
        reference: StructureReference,
        probability: f32,
    },
    Loop { // Create "count" instances of the referenced structure, positioned by applying index x shift_transform, and each child having applied index x child_transform
        reference: StructureReference,
        shift_transform: EulerTransform,
        child_transform: EulerTransform,
        count: usize,
    },
    NestingLoop{
        reference: StructureReference,
        repeated_transform: EulerTransform,
        count: usize,
    },
    NoiseSpawn {
        reference: StructureReference,
        fbm: FBMData,
        sample_size: SampleSize,
        count: u32,
        exclusivity_radius:f32,
        resolution_modifier:f32
    },
    PathSpawn{  //Spawn along a path. I would like to add the ability to jiggle the points.
        reference: StructureReference,
        points: Vec<Vec3>,
        tension: f32,
        spread: SpreadData,
        count: u32
    },
    Reflection{
        reference: StructureReference,
        reflection_plane: Plane3d,
        reflection_point: Vec3,
        reflect_child: bool
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ColliderType {
    None,
    //Compound{},
    Ball{
        radius: f32
    },
    Cylinder{
        half_height: f32, radius: f32
    },
    RoundCylinder{
        half_height: f32, radius: f32, border_radius: f32
    },
    Cone{
        half_height: f32, radius: f32
    },
    RoundCone{
        half_height: f32, radius: f32, border_radius: f32
    },
    Capsule{
        start: Vec3, end: Vec3, radius: f32,
    },
    CapsuleX{
        half_height: f32, radius: f32,
    },
    CapsuleY{
        half_height: f32, radius: f32,
    },
    CapsuleZ{
        half_height: f32, radius: f32,
    },
    Cuboid{
        hx: f32, hy: f32, hz: f32
    },
    RoundCuboid{
        half_x: f32, half_y: f32, half_z: f32, border_radius: f32
    },
    Segment{
        a: Vec3, b: Vec3
    },
    Triangle{
        a: Vec3, b: Vec3, c: Vec3
    },
    RoundTriangle{
        a: Vec3, b: Vec3, c: Vec3, border_radius: f32
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StructureReference {
    Raw(Box<Structure>),
    Ref(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RandData {
    Linear(f32),
    Gaussian(f32),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SpreadData {
    Regular,
    Gaussian(f32),
    Noise{
        fbm_data: FBMData,
        sample_size : f32,
        exclusivity_radius : f32,
        resolution_modifier : f32
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SampleSize {
    UUniDim(i32),
    UBiDim(i32),
    UTriDim(i32),
    UniDim(i32),
    BiDim(i32,i32),
    TriDim(i32,i32,i32)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FBMData {
    seed: SeededOrNot,
    scale: f32,
    octaves: u8,
    frequency: f32,
    lacunarity: f32,
    persistence: f32
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SeededOrNot {
    Seeded(u64),
    Unseeded
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Structure {
    pub data: Vec<(StructureKey, EulerTransform)>,
}

impl Structure {
    pub fn create_random_substructure(&self, n: &usize, rng: &mut WyRand) -> Self {
        let selected_data = if *n >= self.data.len() {
            self.data.clone()
        } else {
            self.data
                .choose_multiple(rng, *n)
                .cloned()
                .collect::<Vec<_>>()
        };

        Structure {
            data: selected_data,
        }
    }
}

impl<'a> TryFrom<&'a StructureReference> for Structure {
    type Error = StructureError;

    fn try_from(value: &'a StructureReference) -> Result<Self, Self::Error> {
        match value {
            StructureReference::Raw(boxed_structure) => {
                Ok(boxed_structure.as_ref().clone())
            },
            StructureReference::Ref(name) => {
                import_structure(name.clone())
                    .map_err(|e| StructureError::ImportFailed(e.to_string()))
            }
        }
    }
}

pub fn import_structure(structure_name: String) -> Result<Structure, ron::Error> {
    let file_path = format!("assets/structures/{}.arch", structure_name);
    let file = File::open(&file_path)?;
    let deserialized: Result<Structure, SpannedError> = from_reader(file);
    deserialized.map_err(|e| e.into())
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "PointLight")]
pub struct SerializablePointLight {
    pub color: Color,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub shadows_enabled: bool,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "SpotLight")]
pub struct SerializableSpotLight {
    pub color: Color,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub shadows_enabled: bool,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
    pub outer_angle: f32,
    pub inner_angle: f32,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "DirectionalLight")]
pub struct SerializableDirectionalLight {
    pub color: Color,
    pub illuminance: f32,
    pub shadows_enabled: bool,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "AmbientLight")]
pub struct SerializableAmbientLight{
    pub color: Color,
    pub brightness: f32,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "FogSettings")]
pub struct SerializableFogSettings {
    pub color: Color,
    pub directional_light_color: Color,
    pub directional_light_exponent: f32,
    #[serde(with = "SerializableFogFalloff")]
    pub falloff: FogFalloff,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "FogFalloff")]
pub enum SerializableFogFalloff {
    Linear {
        start: f32,
        end: f32,
    },
    Exponential {
        density: f32,
    },
    ExponentialSquared {
        density: f32,
    },
    Atmospheric {
        extinction: Vec3,
        inscattering: Vec3,
    },
}

#[derive(Debug)]
pub enum StructureError {
    CycleDetected(String),
    ImportFailed(String),
    Other(String), // You can add more specific errors as needed
}

pub(crate) fn spawn_mesh(
    commands: &mut Commands,
    material_cache: &Res<MaterialCache>,
    meshes: &mut ResMut<Assets<Mesh>>,
    mesh: &Mesh,
    transform: Transform,
    material: &TMaterial,
)
{
    let (material_name, adjusted_mesh) = match material {
        TMaterial::BasicMaterial { material_name } => {
            (material_name.clone(), mesh.clone()) // No tiling factor adjustment needed
        },
        TMaterial::TiledMaterial { material_name, tiling_factor } => {
            let mut mesh = mesh.clone();
            if let Some(bevy::render::mesh::VertexAttributeValues::Float32x2(uvs)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
                for uv in uvs.iter_mut() {
                    uv[0] *= tiling_factor.x;
                    uv[1] *= tiling_factor.y;
                }
            }
            (material_name.clone(), mesh)
        },
    };

    if let Some(material_handle) = material_cache.get(&material_name) {
        let mesh_handle = meshes.add(adjusted_mesh);
        commands.spawn(PbrBundle {
            mesh: mesh_handle,
            transform,
            material: material_handle.clone(),
            ..default()
        })
            .insert(Name::new("Floor"))
            .insert(Collider::halfspace(Vect::Y).expect("Didn't work!"))
            .insert(RigidBody::KinematicPositionBased)
            .insert(ActiveEvents::CONTACT_FORCE_EVENTS);
    } else {
        println!("Material not found: {}", material_name);
    }
}

pub(crate) fn spawn_structure_by_name(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    structure_name: String,
    parent_transform: Transform,
    struct_stack: &mut Vec<String>,
    spawn_queue: &mut Vec<(StructureKey, Transform)>,
    gen_rng: &mut ResMut<GenRng>,
    dir_light_writer: &mut EventWriter<DirLightEvent>,
    amb_light_writer: &mut EventWriter<AmbLightEvent>,
    fog_writer: &mut EventWriter<FogEvent>,
    music_writer: &mut EventWriter<BGMusicEvent>,
    sfx_writer: &mut EventWriter<SFXEvent>
) -> Result<(), StructureError>
{
    // Check for cycle detection in the struct_stack.
    if struct_stack.contains(&structure_name) {
        return Err(StructureError::CycleDetected(structure_name));
    }

    // Check if the recursive depth exceeds the maximum allowed depth
    if struct_stack.len() >= 100 {
        return Err(StructureError::Other(format!("Maximum recursion depth exceeded while processing {}", structure_name)));
    }

    struct_stack.push(structure_name.clone());

    // Attempt to import the structure based on its name.
    let result = match crate::structure::import_structure(structure_name.clone()) {
        Ok(structure) => {
            // Recursively spawn the structure data.
            spawn_structure_by_data(
                commands,
                asset_server,
                &structure,
                parent_transform,
                struct_stack,
                spawn_queue,
                gen_rng,
                dir_light_writer,
                amb_light_writer,
                fog_writer,
                music_writer,
                sfx_writer
            )
        },
        Err(_) => {
            // Handle import failure.
            Err(StructureError::ImportFailed(format!("Can't find file: {:?}", structure_name.clone())))
        }
    };

    // Pop the current structure name off the stack after processing.
    struct_stack.pop();

    // Process the spawn queue only if the struct_stack is empty, indicating all recursions are complete.
    if result.is_ok() && struct_stack.is_empty() {
        // Process all queued spawn commands.
        while let Some((struct_key, transform)) = spawn_queue.pop() {
            match struct_key {
                StructureKey::GLTF{..} => {
                    spawn_scene_from_path(commands, asset_server, &struct_key, transform, Transform::IDENTITY);
                },
                StructureKey::PointLight( light ) => {
                    spawn_point_light(commands, light.into(), transform);
                },
                StructureKey::SpotLight( light ) => {
                    spawn_spot_light(commands, light.into(), transform);
                },
                StructureKey::DirectionalLight( light ) => {
                    dir_light_writer.send(DirLightEvent::SetDirLight {
                        light,
                        transform,
                    });
                }
                StructureKey::AmbientLight( light ) => {
                    amb_light_writer.send(AmbLightEvent::SetAmbLight {
                        light
                    });
                }
                StructureKey::FogSettings( fog ) => {
                    fog_writer.send(FogEvent::SetFog {
                        fog
                    });
                }
                StructureKey::BackgroundMusic( filepath ) => {
                    music_writer.send(BGMusicEvent::SetBGMusic {
                        filepath
                    });
                }
                StructureKey::SoundEffect(filepath ) => {
                    sfx_writer.send(SFXEvent::CreateAudioEmitter {
                        filepath, transform
                    });
                }
                _ => {}
            }
        }
    }

    // Pass the result of the data processing upwards.
    result
}

pub(crate) fn spawn_point_light(
    commands: &mut Commands,
    point_light: PointLight,
    transform: Transform
)
{
    let entity = commands.spawn(PointLightBundle {
        point_light,
        transform,
        ..Default::default()
    }).id();
    commands.entity(entity).insert(Name::new("Pointlight".to_string()));
}

pub(crate) fn spawn_spot_light(
    commands: &mut Commands,
    spot_light: SpotLight,
    transform: Transform
)
{
    let entity = commands.spawn(SpotLightBundle {
        spot_light,
        transform,
        ..Default::default()
    }).id();
    commands.entity(entity).insert(Name::new("Spotlight".to_string()));
}

impl From<&str> for StructureError {
    fn from(error: &str) -> Self {
        StructureError::Other(error.to_string())
    }
}

pub(crate) fn spawn_structure_by_data(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    structure: &Structure,
    parent_transform: Transform,
    struct_stack: &mut Vec<String>,
    spawn_queue: &mut Vec<(StructureKey, Transform)>,
    gen_rng: &mut ResMut<GenRng>,
    dir_light_writer: &mut EventWriter<DirLightEvent>,
    amb_light_writer: &mut EventWriter<AmbLightEvent>,
    fog_writer: &mut EventWriter<FogEvent>,
    music_writer: &mut EventWriter<BGMusicEvent>,
    sfx_writer: &mut EventWriter<SFXEvent>
) -> Result<(), StructureError>
{
    for (key, local_transform) in &structure.data {
        let combined_transform = parent_transform * Transform::from(local_transform.clone());
        match key {
            StructureKey::GLTF {..} => {
                spawn_queue.push((key.clone() , combined_transform));
            },
            StructureKey::Nest ( reference ) => {
                let struc = Structure::try_from(reference)?;
                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &struc,
                    combined_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer,
                )?;
            }
            StructureKey::PointLight ( _ ) => {
                spawn_queue.push((key.clone(), combined_transform));
            },
            StructureKey::ProbabilitySpawn { reference, probability } => {
                if gen_rng.rng_mut().gen::<f32>() < *probability {
                    let struc = Structure::try_from(reference)?;
                    spawn_structure_by_data(
                        commands,
                        asset_server,
                        &struc,
                        combined_transform,
                        struct_stack,
                        spawn_queue,
                        gen_rng,
                        dir_light_writer,
                        amb_light_writer,
                        fog_writer,
                        music_writer,
                        sfx_writer
                    )?;
                }
            }
            StructureKey::Choose { list } => {
                let struc = Structure::try_from(list)?;
                let sub_struc = struc.create_random_substructure(&(1usize), gen_rng.rng_mut());
                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &sub_struc,
                    combined_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            },
            StructureKey::ChooseSome { list, count: num } => {
                let struc = Structure::try_from(list)?;
                let sub_struc = struc.create_random_substructure(num, gen_rng.rng_mut());
                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &sub_struc,
                    combined_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            },
            StructureKey::Loop { reference, shift_transform, child_transform, count } => {
                let positions = get_looped_position_list(combined_transform.translation, shift_transform.clone().into(), *count);

                println!("{:?}", child_transform);

                let child_transforms: Vec<EulerTransform> = (0..*count).map(|n| {
                    EulerTransform {
                        translation: (child_transform.translation.0 * n as f32, child_transform.translation.1 * n as f32, child_transform.translation.2 * n as f32),
                        rotation: (child_transform.rotation.0 * n as f32, child_transform.rotation.1 * n as f32, child_transform.rotation.2 * n as f32),
                        scale: (1.0 + child_transform.scale.0 * n as f32, 1.0 + child_transform.scale.1 * n as f32, 1.0 + child_transform.scale.2 * n as f32)
                    }
                }).collect();

                let structure = Structure {
                    data: positions.into_iter().zip(child_transforms.iter())
                        .map(|(pos, transform)| {
                            let euler_transform = EulerTransform {
                                translation: (pos.x + transform.translation.0, pos.y + transform.translation.1, pos.z + transform.translation.2),
                                rotation: (transform.rotation.0, transform.rotation.1, transform.rotation.2),
                                scale: (transform.scale.0, transform.scale.1, transform.scale.2)
                            };
                            println!("{:?}", euler_transform);

                            (StructureKey::Nest ( reference.clone() ), euler_transform)
                        })
                        .collect()
                };

                spawn_structure_by_data(commands,
                                        asset_server,
                                        &structure,
                                        combined_transform,
                                        struct_stack,
                                        spawn_queue,
                                        gen_rng,
                                        dir_light_writer,
                                        amb_light_writer,
                                        fog_writer,
                                        music_writer,
                                        sfx_writer)?;
            }
            StructureKey::DirectionalLight( _) => {
                spawn_queue.push((key.clone() , Transform::from(local_transform.clone())));
                }
            StructureKey::AmbientLight(_) => {
                spawn_queue.push((key.clone() , Transform::from(local_transform.clone())));
            }
            StructureKey::FogSettings(_) => {
                spawn_queue.push((key.clone() , Transform::from(local_transform.clone())));
            }
            StructureKey::BackgroundMusic(_) => {
                spawn_queue.push((key.clone() , Transform::from(local_transform.clone())));
            }
            StructureKey::SoundEffect(_)  => {
                spawn_queue.push((key.clone() , combined_transform));
            }
            StructureKey::SpotLight(_) => {
                spawn_queue.push((key.clone() , combined_transform));
            }
            StructureKey::Rand{reference, rand} => {
                let jiggled = jiggle_transform(gen_rng, rand.clone(), local_transform.clone());

                let structure = Structure{ data: vec![(StructureKey::Nest ( reference.clone() ), jiggled)]};

                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &structure,
                    parent_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            }
            StructureKey::NoiseSpawn { reference, .. } => {

                let points = generate_noise_spawn_points(key, gen_rng);

                let struc_data: Vec<(StructureKey, EulerTransform)> = points.iter().map(|(x, y, z)| {
                    // Calculate the new translation based on the provided formula
                    let new_translation = Vec3::new(
                        local_transform.translation.0 + local_transform.scale.0**x,
                        local_transform.translation.1 + local_transform.scale.1**z,
                        local_transform.translation.2 + local_transform.scale.2**y,
                    );

                    // Create a new transform with the calculated translation, zero rotation, and unit scale
                    let new_transform = EulerTransform {
                        translation: (new_translation.x, new_translation.y, new_translation.z),
                        rotation: (0.0,0.0,0.0), // Zero rotation
                        scale: (1.0,1.0,1.0),     // Unit scale
                    };

                    (StructureKey::Nest(reference.clone()), new_transform) // Assuming you need to create a new StructureKey here
                }).collect();

                let structure_unwrapped = Structure{ data: struc_data };

                // Wrap the structure_unwrapped in a Raw Nesting
                let raw_nesting = StructureReference::Raw(Box::new(structure_unwrapped));

                // Use the wrapped raw nesting as the reference in a Nest StructureKey
                let structure = Structure {
                    data: vec![
                        (StructureKey::Nest(raw_nesting), parent_transform.into())
                    ],
                };

                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &structure,
                    parent_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            }
            StructureKey::PathSpawn { reference, points, tension, spread, count } => {
                let curve = CubicCardinalSpline::new(*tension, points.clone()).to_curve();
                let positions: Vec<Vec3> = match spread {
                    SpreadData::Regular => {
                        curve.iter_positions(*count as usize).collect()
                    },
                    _ => {
                        panic!("This spread type not supported yet!");
                    },
                };

                let struc_data: Vec<(StructureKey, EulerTransform)> = positions.iter().map(|point| {
                    let euler_transform = EulerTransform {
                        translation: (point.x, point.y, point.z),
                        rotation: (0.0, 0.0, 0.0),
                        scale: (1.0, 1.0, 1.0),
                    };

                    (StructureKey::Nest(reference.clone()), euler_transform)
                    //
                }).collect();

                let structure = Structure {
                    data: struc_data
                };

                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &structure,
                    parent_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            }
            StructureKey::Reflection { reference, reflection_plane, reflection_point, reflect_child } => {

                if *reflect_child {
                    return Err("Child reflection not implemented!".into());
                }

                let reflected_location = reflect_point(
                    Vec3::new(local_transform.translation.0, local_transform.translation.1, local_transform.translation.2),
                    *reflection_plane,
                    *reflection_point);

                let mut reflected_transform = local_transform.clone();
                reflected_transform.translation = (reflected_location.x, reflected_location.y, reflected_location.z);

                let reflected_combined_transform = parent_transform * Transform::from(reflected_transform);

                println!("{:?}",local_transform);
                println!("{:?}", reflected_location);

                let structure_internal = match Structure::try_from(reference) {
                    Ok(structure) => structure,
                    Err(error) => {
                        return Err(error);
                    }
                };

                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &structure_internal,
                    combined_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;

                spawn_structure_by_data(
                    commands,
                    asset_server,
                    &structure_internal,
                    reflected_combined_transform,
                    struct_stack,
                    spawn_queue,
                    gen_rng,
                    dir_light_writer,
                    amb_light_writer,
                    fog_writer,
                    music_writer,
                    sfx_writer
                )?;
            }
            StructureKey::NestingLoop { reference, repeated_transform, count } => {
                //IF COUNT>0
                if *count>0
                {
                    //BUILD IT USING COMBINEDTRANSFORM
                    spawn_structure_by_data(
                        commands,
                        asset_server,
                        &(Structure::try_from(reference).expect("What did you do??")),
                        combined_transform,
                        struct_stack,
                        spawn_queue,
                        gen_rng,
                        dir_light_writer,
                        amb_light_writer,
                        fog_writer,
                        music_writer,
                        sfx_writer
                    )?;

                    //BUILD THE NEXT ONE USING THE SAME NESTINGLOOP WITH COUNT-1
                    let new_loop = StructureKey::NestingLoop{
                        reference: reference.clone(),
                        repeated_transform: repeated_transform.clone(),
                        count: *count-1
                    };

                    let structure = Structure{
                        data: vec![(new_loop,repeated_transform.clone())]
                    };

                    spawn_structure_by_data(
                        commands,
                        asset_server,
                        &structure,
                        combined_transform,
                        struct_stack,
                        spawn_queue,
                        gen_rng,
                        dir_light_writer,
                        amb_light_writer,
                        fog_writer,
                        music_writer,
                        sfx_writer
                    )?;
                }

            }
        }
    }

    Ok(())
}

pub(crate) fn reflect_point(
    point: Vec3,
    reflection_plane: Plane3d,
    reflection_point: Vec3,
) -> Vec3 {
    // Compute the vector from the reflection point to the point
    let point_to_reflection_point = point - reflection_point;

    let norm_as_vec = Vec3::new(reflection_plane.normal.x, reflection_plane.normal.y, reflection_plane.normal.z);

    // Project this vector onto the plane's normal
    let projection = point_to_reflection_point.dot(*reflection_plane.normal) * norm_as_vec;

    // Reflect the point across the plane
    point - 2.0 * projection
}

pub (crate)fn jiggle_transform(
    gen_rng: &mut ResMut<GenRng>,
    rand_data: RandData,
    original_transform: EulerTransform,
) -> EulerTransform
{
    let random_floats: Vec<f32> = match rand_data {
        RandData::Linear(spread)  => {
           (0..7).map(|_| gen_rng.rng_mut().gen::<f32>() * spread - spread/2.0).collect()
        }
        RandData::Gaussian(standard_deviation)  => {
            let normal_dist = Normal::new(0.0, standard_deviation as f64).unwrap();
            (0..7).map(|_| normal_dist.sample(gen_rng.rng_mut()) as f32).collect()
        }
    };

    EulerTransform {
        translation: (
            original_transform.translation.0 * random_floats[0],
            original_transform.translation.1 * random_floats[1],
            original_transform.translation.2 * random_floats[2],
        ),
        rotation: (
            original_transform.rotation.0 * random_floats[3],
            original_transform.rotation.1 * random_floats[4],
            original_transform.rotation.2 * random_floats[5],
        ),
        scale: ( //Scale these all together for now
                 2.0f32.powf(original_transform.scale.0 *random_floats[6]),
                 2.0f32.powf(original_transform.scale.1 *random_floats[6]),
                 2.0f32.powf(original_transform.scale.2 *random_floats[6])
        ),
    }
}

pub(crate) fn spawn_scene_from_path(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    model_data: &StructureKey,
    global_transform: Transform,
    local_transform: Transform
)
{
    if let StructureKey::GLTF { path, priority, collider } = model_data {

        //Want to replace this with the preloaded collection
        let scene_handle: Handle<Scene> = asset_server.load(path);

        let entity = commands.spawn(SceneBundle {
            scene: scene_handle,
            transform: global_transform * local_transform,
            ..Default::default()
        }).id();

        let filename = Path::new(path)
            .file_name()
            .and_then(|file_name| file_name.to_str()).unwrap();

        commands.entity(entity).insert(Name::new(format!("{:?}", filename)));

        if let Some(_) = create_collider(collider) {
            commands.entity(entity)
                .insert(RigidBody::Dynamic)
                .insert(Dominance::group(*priority))
                .insert(Damping { linear_damping:  1.0, angular_damping: 0.0 })
                .insert(LockedAxes::ROTATION_LOCKED | LockedAxes::TRANSLATION_LOCKED_Y)
                .insert(ActiveEvents::CONTACT_FORCE_EVENTS)
                .insert(Sleeping{
                    normalized_linear_threshold: 0.01,
                    angular_threshold: 0.01,
                    sleeping: false,
                })
                .insert(ContactForceEventThreshold(0.0));
        }
    }
}

#[derive(Component)]
pub(crate) struct MainCamera;
#[derive(Component)]
pub(crate) struct MainDirectionalLight;

pub(crate) fn fog_updater_system(
    mut update_reader: EventReader<FogEvent>,
    mut fog_query: Query<&mut FogSettings, With<MainCamera>>,
) {
    for event in update_reader.read() {
        match event {
            FogEvent::SetFog{fog} => {
                for mut fog_settings in fog_query.iter_mut() {
                    *fog_settings = fog.clone();
                }
            }
        }
    }
}

pub(crate) fn directional_light_updater_system(
    mut update_reader: EventReader<DirLightEvent>,
    mut light_query: Query<(&mut DirectionalLight, &mut Transform), With<MainDirectionalLight>>,
) {
    for event in update_reader.read() {
        match event {
            DirLightEvent::SetDirLight{light: new_light, transform: new_transform} => {
                for (mut light, mut transform) in light_query.iter_mut() {
                    *light = new_light.clone();
                    *transform = new_transform.clone();
                }
            }
        }
    }
}

pub(crate) fn ambient_light_updater_system(
    mut update_reader: EventReader<AmbLightEvent>,
    mut ambient_light: ResMut<AmbientLight>,
) {
    for event in update_reader.read() {
        match event {
            AmbLightEvent::SetAmbLight{light} => {
                *ambient_light = light.clone();
            }
        }
    }
}

pub(crate) fn object_spawn_reader_system(
    mut spawn_reader: EventReader<ObjectSpawnEvent>,
    asset_server: Res<AssetServer>,
    material_cache: Res<MaterialCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut gen_rng: ResMut<GenRng>,
    mut commands: Commands,
    mut dir_light_writer: EventWriter<DirLightEvent>,
    mut amb_light_writer: EventWriter<AmbLightEvent>,
    mut fog_writer: EventWriter<FogEvent>,
    mut music_writer: EventWriter<BGMusicEvent>,
    mut sfx_writer: EventWriter<SFXEvent>
) {
    for spawn_event in spawn_reader.read() {
        match spawn_event {
            ObjectSpawnEvent::MeshSpawn { mesh, transform, material } => {
                spawn_mesh(
                    &mut commands,
                    &material_cache,
                    &mut meshes,
                    mesh,
                    Transform::from(transform.clone()),
                    &material
                );
            },

            ObjectSpawnEvent::SceneSpawn { data, transform } => {
                spawn_scene_from_path(&mut commands, &asset_server, data, Transform::from(transform.clone()), Transform::IDENTITY);
            }
            ObjectSpawnEvent::StructureSpawn { structure, transform } => {
                match spawn_structure_by_name(
                    &mut commands,
                    &asset_server,
                    structure.clone(),
                    Transform::from(transform.clone()),
                    &mut vec![],
                    &mut vec![],
                    &mut gen_rng,
                    &mut dir_light_writer,
                    &mut amb_light_writer,
                    &mut fog_writer,
                    &mut music_writer,
                    &mut sfx_writer
                ){
                    Ok(_) => {},
                    Err(error) => {
                        eprintln!("Failed to spawn structure: {} because {:?}", structure, error);
                    }
                }
            }
        }
    }
}

fn get_looped_position_list(origin: Vec3, transform: EulerTransform, x_times: usize) -> Vec<Vec3> {
    let mut positions = Vec::new();

    for n in 1..=x_times {
        // Calculate the translated and scaled position increment
        let translated = Vec3::new(
            transform.translation.0 * (1.0 + transform.scale.0 * (n as f32 - 1.0)),
            transform.translation.1 * (1.0 + transform.scale.1 * (n as f32 - 1.0)),
            transform.translation.2 * (1.0 + transform.scale.2 * (n as f32 - 1.0)),
        );

        // Convert Euler angles to quaternion for rotation
        let rotation = Quat::from_euler(
            bevy::math::EulerRot::XYZ,
            transform.rotation.0.to_radians() * (n - 1) as f32,
            transform.rotation.1.to_radians() * (n - 1) as f32,
            transform.rotation.2.to_radians() * (n - 1) as f32,
        );

        // Apply rotation to the translated vector
        let rotated_pos = rotation * translated;

        // Push the new position to the list
        positions.push(origin+rotated_pos);
    }

    positions
}

fn generate_noise_spawn_points(
    data: &StructureKey,
    gen_rng: &mut ResMut<GenRng>,
) -> Vec<(f32, f32, f32)>
{
    let (fbm, sample_size, count, exclusivity_radius, resolution_modifier) = if let StructureKey::NoiseSpawn {
        fbm,
        sample_size,
        count,
        exclusivity_radius,
        resolution_modifier, ..
    } = data {
        (fbm, sample_size, count, exclusivity_radius, resolution_modifier)
    } else {
        unreachable!()
    };

    let seed = match fbm.seed{
        SeededOrNot::Seeded(s) => {s}
        SeededOrNot::Unseeded => {gen_rng.rng_mut().gen::<u64>()}
    };

    match sample_size {
        SampleSize::UBiDim(x) => {
            generate_noise_spawn_points_2d(
                (x,x),
                fbm.scale,
                fbm.octaves,
                fbm.frequency,
                fbm.lacunarity,
                fbm.persistence,
                count,
                exclusivity_radius,
                resolution_modifier,
                seed,
            )
        }
        SampleSize::BiDim(x, y) => {
            generate_noise_spawn_points_2d(
                (x,y),
                fbm.scale,
                fbm.octaves,
                fbm.frequency,
                fbm.lacunarity,
                fbm.persistence,
                count,
                exclusivity_radius,
                resolution_modifier,
                seed,
            )
        }
        _ => {
            panic!("THIS NOISE DIMENSIONALITY IS NOT IMPLEMENTED!");
        }
    }
}

/// Generates a filtered noise map based on the provided parameters.
///
/// # Arguments
/// * `sample_size` - (width, height) of the sampling area.
/// * `scale` - Scale for the noise.
/// * `octaves` - Number of octaves for the noise.
/// * `frequency` - Base frequency of the noise.
/// * `lacunarity` - Lacunarity of the noise.
/// * `persistence` - Persistence of the noise.
/// * `spawn_count` - Number of points to spawn.
/// * `exclusivity_radius` - Radius to filter exclusive points.
/// * `resolution_modifier` - Modifier for the resolution.
///
/// # Returns
/// A vector of tuples, each containing the x and y coordinates, and the noise value.
pub fn generate_noise_spawn_points_2d(
    sample_size: (&i32, &i32),
    scale: f32,
    octaves: u8,
    frequency: f32,
    lacunarity: f32,
    persistence: f32,
    spawn_count: &u32,
    exclusivity_radius: &f32,
    resolution_modifier: &f32,
    seed: u64,
) -> Vec<(f32, f32, f32)> {
    // Sanity checks
    assert!(octaves >= 1 && octaves <= 6, "Octaves must be between 1 and 6.");
    assert!(*sample_size.0 > 0 && *sample_size.1 > 0, "Sample size dimensions must be positive.");
    assert!(*resolution_modifier > 0.0, "Resolution modifier must be positive.");

    let effective_width = *sample_size.0 as f32 * resolution_modifier;
    let effective_height = *sample_size.1 as f32 * resolution_modifier;

    assert!(effective_width as i64 * effective_height as i64<= 2097152,
            "Product of sample size dimensions and resolution modifier must be no larger than 262144.");
    assert!(effective_width%2.0 == 0.0 && effective_height%2.0 == 0.0,
            "Effective dimensions (sample size multiplied by resolution modifier) must result in integers divisble by 2.");

    let generator = Source::simplex(seed)
        .fbm(octaves as u32, frequency as f64, lacunarity as f64, persistence as f64)
        .scale([scale as f64; 3]);

    let sample_size = sample_size;
    let resolution_modifier = resolution_modifier;

    let mut values_and_coords = Vec::new();

    //NOTE THAT ALL STARTS AND ENDS ARE OFFSET. THIS AVOIDS SOME NOISE ARTIFACTING AROUND THE NOISE ZERO
    let start_sample_x = sample_size.0/2 * *resolution_modifier as i32;
    let end_sample_x = 3 * sample_size.0/2 * *resolution_modifier as i32;
    let start_sample_y = sample_size.1/2 * *resolution_modifier as i32;
    let end_sample_y = 3 * sample_size.1/2 * *resolution_modifier as i32;

    let radius_x = (end_sample_x - start_sample_x) as f32 / 2.0;
    let radius_y = (end_sample_y - start_sample_y) as f32 / 2.0;

    //let image_width = ((end_sample_x - start_sample_x) as f32 * resolution_modifier) as u32+1;
    //let image_height = ((end_sample_y - start_sample_y) as f32 * resolution_modifier) as u32+1;
    //let mut img = image::GrayImage::new(image_width, image_height);

    let centerpoint = Vec2::new(
        ((start_sample_x+end_sample_x)/2) as f32,
        ((start_sample_y+end_sample_y)/2) as f32
    );

    for x in start_sample_x..=end_sample_x {
        for y in start_sample_y..=end_sample_y {
            let sample_x = x as f32 / resolution_modifier;
            let sample_y = y as f32 / resolution_modifier;

            //let img_pos = (
            //    ((x-start_sample_x) as f32 * resolution_modifier) as u32,
            //    ((y-start_sample_y) as f32 * resolution_modifier) as u32);

            //THIS BLOCK DISMISSES ANY POINTS OUTSIDE THE CIRCUMSCRIBED ELLIPSOID
            let nx = (sample_x - centerpoint.x) / radius_x;
            let ny = (sample_y - centerpoint.y) / radius_y;

            // Exclude points outside the normalized unit circle
            if nx.powi(2) + ny.powi(2) > 1.0 {
                continue;
            }
            //

            let value = generator.sample([sample_x as f64, sample_y as f64, (start_sample_x as f64 + start_sample_y as f64)/2.0]);

            //let normalized_value = ((value + 1.0) / 2.0 * 255.0) as u8; // Normalize and scale to 0-255
            //img.put_pixel(img_pos.0, img_pos.1, image::Luma([normalized_value]));

            values_and_coords.push((sample_x, sample_y, 0.0, value));
        }
    }

    // Sorting by value in descending order
    values_and_coords.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    // Filtering exclusive results based on `exclusivity_radius` and `spawn_count`
    let filtered_results = filter_by_exclusivity(&values_and_coords, spawn_count, exclusivity_radius);

    //img.save("noise_slice.png").expect("Failed to save image.");

    let normalized_points: Vec<(f32, f32, f32)> = filtered_results.iter()
        .map(|&(x, y, _)| {
            (
                (x - centerpoint.x) / centerpoint.x,
                (y - centerpoint.y) / centerpoint.y,
                0.0 // normalized_z as a constant 0.0
            )
        })
        .collect();

    //println!("Selected points: ({:?})", normalized_points);

    normalized_points
}

fn filter_by_exclusivity(
    sorted_values: &Vec<(f32, f32, f32, f64)>, // (x, y, z, value)
    n: &u32,
    radius: &f32,
) -> Vec<(f32, f32, f32)> {
    let mut results = Vec::new();
    let mut candidates = std::collections::VecDeque::from(sorted_values.to_vec());

    let square_radius = radius*radius;

    while results.len() < *n as usize && !candidates.is_empty() {

        if let Some((x, y, z, _)) = candidates.pop_front() {
            // Add current element to results
            results.push((x, y, z));

            // Filter out all candidates within R of the accepted element
            candidates = candidates.into_iter().filter(|&(cx, cy, cz, _)| {
                let square_distance = (x - cx).powi(2) + (y - cy).powi(2)+ (z - cz).powi(2);
                square_distance > square_radius
            }).collect();

            // If we've collected enough results, break early
            if results.len() >= *n as usize {
                break;
            }
        }
    }

    results
}

#[allow(dead_code)]
#[derive(Event)]
pub enum ObjectSpawnEvent {
    MeshSpawn {
        mesh: Mesh,
        transform: EulerTransform,
        material: TMaterial,
    },
    SceneSpawn {
        data: StructureKey,
        transform: EulerTransform,
    },
    StructureSpawn
    {
        structure: String,
        transform: EulerTransform,
    }
}

#[allow(dead_code)]
#[derive(Event)]
pub enum FogEvent {
    SetFog{
        fog: FogSettings
    }
}

#[allow(dead_code)]
#[derive(Event)]
pub enum DirLightEvent {
    SetDirLight {
        light: DirectionalLight,
        transform: Transform,
    }
}

#[allow(dead_code)]
#[derive(Event)]
pub enum AmbLightEvent {
    SetAmbLight{
        light: AmbientLight
    }
}

#[allow(dead_code)]
#[derive(Event)]
pub enum BGMusicEvent {
    SetBGMusic{
        filepath: String
    }
}

#[allow(dead_code)]
#[derive(Event)]
pub enum SFXEvent {
    CreateAudioEmitter{
        filepath: String,
        transform: Transform
    }
}

#[allow(dead_code)]
pub enum TMaterial
{
    BasicMaterial {
        material_name: String,
    },
    TiledMaterial {
        material_name: String,
        tiling_factor: Vec2,
    },
}

fn create_collider(collider_type: &ColliderType) -> Option<Collider> {
    match collider_type {
        ColliderType::None => None,
        ColliderType::Ball { radius } => Some(Collider::ball(*radius)),
        ColliderType::Cylinder { half_height, radius } => Some(Collider::cylinder(*half_height, *radius)),
        ColliderType::RoundCylinder { half_height, radius, border_radius } =>
            Some(Collider::round_cylinder(*half_height, *radius, *border_radius)),
        ColliderType::Cone { half_height, radius } => Some(Collider::cone(*half_height, *radius)),
        ColliderType::RoundCone { half_height, radius, border_radius } =>
            Some(Collider::round_cone(*half_height, *radius, *border_radius)),
        ColliderType::Capsule { start, end, radius } => Some(Collider::capsule(*start, *end, *radius)),
        ColliderType::CapsuleX { half_height, radius } => Some(Collider::capsule_x(*half_height, *radius)),
        ColliderType::CapsuleY { half_height, radius } => Some(Collider::capsule_y(*half_height, *radius)),
        ColliderType::CapsuleZ { half_height, radius } => Some(Collider::capsule_z(*half_height, *radius)),
        ColliderType::Cuboid { hx, hy, hz } => Some(Collider::cuboid(*hx, *hy, *hz)),
        ColliderType::RoundCuboid { half_x, half_y, half_z, border_radius } =>
            Some(Collider::round_cuboid(*half_x, *half_y, *half_z, *border_radius)),
        ColliderType::Segment { a, b } => Some(Collider::segment(*a, *b)),
        ColliderType::Triangle { a, b, c } => Some(Collider::triangle(*a, *b, *c)),
        ColliderType::RoundTriangle { a, b, c, border_radius } =>
            Some(Collider::round_triangle(*a, *b, *c, *border_radius)),
    }
}