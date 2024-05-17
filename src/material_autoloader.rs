//DON'T WORRY ABOUT ANYTHING IN HERE. I've removed all assets it would otherwise process, it's just a
//lot of code rewriting in the Structure class to remove it all.

use bevy::{
    prelude::*,
    reflect::Reflect,
};
use bevy::asset::processor::LoadAndSave;
use bevy::pbr::StandardMaterial;
use bevy::render::texture::{CompressedImageSaver, Image};
use bevy::utils::hashbrown::{HashMap};
use bevy_asset_loader::asset_collection::AssetCollection;

pub struct MaterialAutoloader;

#[derive(Resource)]
pub(crate) struct MaterialCache {
    map: HashMap<String, Handle<StandardMaterial>>,
}

impl MaterialCache {
    pub fn new() -> Self {
        MaterialCache {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, handle: Handle<StandardMaterial>) {
        self.map.insert(name, handle);
    }

    pub fn get(&self, name: &str) -> Option<&Handle<StandardMaterial>> {
        self.map.get(name)
    }
}

#[derive(AssetCollection, Resource, Default, Reflect)]
#[reflect(Resource)]
pub(crate) struct MaterialTextures {
    #[asset(path = "materials", collection(typed, mapped))]
    pub(crate) textures: bevy::utils::HashMap<String, Handle<Image>>,
}

impl Plugin for MaterialAutoloader {
    fn build(&self, app: &mut App) {
        let processor: LoadAndSave::<bevy::render::texture::ImageLoader, CompressedImageSaver>  = CompressedImageSaver.into();

        app
            .register_asset_processor(processor)
            .insert_resource(MaterialTextures::default());
    }
}

struct MaterialSets {
    map: HashMap<String, (Option<String>, Option<String>, Option<String>, Option<String>)>,
}

impl MaterialSets {
    pub fn new() -> Self {
        MaterialSets { map: HashMap::new() }
    }
}

pub(crate) fn extract_tex_data(tex_name: &str) -> (String, String) {
    // Define the texture types we are interested in identifying
    let texture_types = ["albedo", "ao", "normal", "met_roughness"];

    // Split the path by '/' to navigate through the path components
    let parts: Vec<&str> = tex_name.split('/').collect();

    // Find the index of "materials" to correctly identify the material's name
    let materials_index = parts.iter().position(|&r| r == "materials").unwrap_or(0);

    // The material name should be right after "materials"
    let material_name = parts.get(materials_index + 1).unwrap_or(&"").to_string();

    // Determine the texture type based on known suffixes
    let texture_type = texture_types.iter()
        .find_map(|&t| {
            if tex_name.contains(&format!("_{}", t)) {
                Some(t.to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    (material_name, texture_type)
}

pub(crate) fn preload_materials_system(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    material_textures: Res<MaterialTextures>,
) {
    let mut material_cache = MaterialCache::new();

    let mut material_sets = MaterialSets::new();

    let tex_paths = material_textures.textures.keys();

    for file_path in tex_paths {
        debug!("Loading material: {}", file_path);

        let (mat_name, tex_type) = extract_tex_data(file_path);
        if tex_type == "unknown" {
            continue;  // Skip if the texture type is not recognized
        }

        // Entry API is used for efficient and clean access or insertion
        let entry = material_sets.map.entry(mat_name.clone()).or_insert_with(|| (None, None, None, None));

        match tex_type.as_str() {
            "albedo" => entry.0 = Some(file_path.to_string()),
            "ao" => entry.1 = Some(file_path.to_string()),
            "normal" => entry.2 = Some(file_path.to_string()),
            "met_roughness" => entry.3 = Some(file_path.to_string()),
            _ => {} // Do nothing if the texture type isn't one of the expected ones
        }

        debug!("Material {} loaded successfully", &mat_name);
    }

    for (material_name, textures) in material_sets.map.iter() {
        let base_tex = textures.0.as_ref().map(|path| asset_server.load(path));
        let ao_tex = textures.1.as_ref().map(|path| asset_server.load(path));
        let normal_tex = textures.2.as_ref().map(|path| asset_server.load(path));
        let met_rough_tex = textures.3.as_ref().map(|path| asset_server.load(path));

        let material_handle = materials.add(StandardMaterial {
            base_color_texture: base_tex,
            occlusion_texture: ao_tex,
            normal_map_texture: normal_tex,
            metallic_roughness_texture: met_rough_tex,
            metallic: 0.1,
            perceptual_roughness: 0.9,
            ..Default::default()
        });

        material_cache.insert(material_name.clone(), material_handle);
    }

    commands.insert_resource(material_cache);
}
