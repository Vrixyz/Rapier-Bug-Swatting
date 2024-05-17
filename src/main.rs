use bevy::prelude::*;
use bevy::input::common_conditions::input_toggle_active;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_rapier3d::prelude::*;

mod material_autoloader;

mod generation;
mod structure;
mod euler_transform;

use material_autoloader::MaterialTextures;

use crate::generation::GeneratorPlugin;
use crate::structure::*;

fn main() {

    let mut app = App::new();

    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                mode: AssetMode::Processed,
                ..default()})
            .build(),
    )
        .add_plugins(
            WorldInspectorPlugin::default().run_if(input_toggle_active(false, KeyCode::Escape)),
        )
        //The MaterialTextures and MaterialPreloader aren't going to do anything for the purpose of reproducing the bug,
        //it's just a lot of code rewrite in the deserialization system to remove them. ingame_setup does, though!
        .register_type::<MaterialTextures>()
        .insert_resource(MaterialTextures{ textures: Default::default() })
        .add_systems(Startup,(material_autoloader::preload_materials_system,ingame_setup).chain())
        //
        .add_event::<ObjectSpawnEvent>()
        .add_event::<FogEvent>()
        .add_event::<DirLightEvent>()
        .add_event::<AmbLightEvent>()
        .add_event::<BGMusicEvent>()
        .add_event::<SFXEvent>()
        .add_plugins(GeneratorPlugin)
        .add_systems(Update,remove_sleeping_bodies).add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default());

    app.run();
}

fn ingame_setup(
    mut commands: Commands,
    spawn_writer: EventWriter<ObjectSpawnEvent>,
) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-10.1, 8.0, 11.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..default()
    });

    generation::generate_map(spawn_writer);
}

pub fn remove_sleeping_bodies(
    mut commands: Commands,
    query: Query<(Entity, &Sleeping, &GlobalTransform)>,
) {
    for (entity, sleeping, trans) in query.iter() {
        if sleeping.sleeping {
            println!("Sleeper found! {:?}", entity);
            commands.entity(entity).despawn_recursive();
        }
        else {
            println!("Moving entity transform {:?} {:?}", entity, trans);
        }
    }
}