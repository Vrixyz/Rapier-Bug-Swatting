use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup_graphics)
        .add_systems(Startup, setup_physics)
        .add_systems(Update, print_position)
        .run();
}

fn setup_graphics(mut commands: Commands) {
    // Add a camera so we can see the debug-render.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(-3.0, 3.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn setup_physics(mut commands: Commands) {
    commands
        .spawn(RigidBody::Dynamic)
        .insert(TransformBundle::from(Transform {
            translation: Vec3::new(0.13110116, 0.0, 1.3211011),
            rotation: Quat::from_xyzw(0.0, -0.33848378, 0.0, 0.94097227),
            scale: Vec3::new(0.9665972, 0.9665972, 0.9665972),
        }));
}

fn print_position(positions: Query<&Transform, With<RigidBody>>) {
    for transform in positions.iter() {
        println!(
            "Body position: {} ; rotation: {}",
            transform.translation, transform.rotation
        );
    }
}
