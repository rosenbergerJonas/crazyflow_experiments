# Safe reach with control barrier functions
A drone flies from a start to a goal behind a field of cylindrical obstacles, commanding attitude
and collective thrust. Control barrier functions keep it away from the obstacles, as a safety filter
on the policy's commands and as reward terms during training.

## Configuration
`config.toml` describes a run. Every table is checked when it is loaded, so misspelled or missing
keys raise an error instead of falling back to a default. Variants of a run are overlay files in
`configs/` that hold only the keys they change, and single values can be overridden on the command
line with their dotted key:

```bash
pixi run -e safe-reach python <script> --overlays dual --drone.dynamics first_principles
```

Overlays are given by their name in `configs/` or as a path to a TOML file, several as a list
(`--overlays "[dual, o3]"`), and apply in that order before the overrides.
