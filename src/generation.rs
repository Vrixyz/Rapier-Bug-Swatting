use bevy::prelude::*;
use bevy::app::{App, Plugin, Update};
use bevy::prelude::{EventWriter};
use bevy_prng::WyRand;

use rand_core::SeedableRng;
use crate::structure::*;

pub struct GeneratorPlugin;

impl Plugin for GeneratorPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(Update,
                         (object_spawn_reader_system,
                          ambient_light_updater_system,
                          directional_light_updater_system,
                          fog_updater_system).chain())
            .insert_resource(GenRng(WyRand::seed_from_u64(124)));
    }
}

pub(crate) fn generate_map(
    mut spawn_writer: EventWriter<ObjectSpawnEvent>,
)
{
    spawn_writer.send(
        ObjectSpawnEvent::StructureSpawn{
            structure: "noise_spawn_obstacles".to_string(),
            transform: Default::default()
        }
    );
}

#[derive(Resource)]
pub(crate) struct GenRng(WyRand);

impl GenRng {
    pub fn rng_mut(&mut self) -> &mut WyRand {
        &mut self.0
    }
}