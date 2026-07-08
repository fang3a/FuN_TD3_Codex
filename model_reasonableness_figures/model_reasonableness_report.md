# Electric-Gas Coupled Microgrid Modeling Reasonableness Report

## 1. Static topology consistency
- Power network bus count: 33 buses.
- Power network branch count: 32 lines; radial tree check: True.
- Gas network junction count: 20 junctions.
- Passive gas pipe count: 21 pipes.
- Compressor count: 2 compressor arcs.
- Gas source nodes: [0, 7].
- Gas load nodes: [2, 5, 6, 9, 11, 14, 15, 18, 19].
- Topology validation result: passed.
- Topology validation warnings: ['Pipe_02_0_1_B is an allowed parallel pipe', 'Pipe_04_1_2_B is an allowed parallel pipe', 'Pipe_11_8_9_B is an allowed parallel pipe', 'Pipe_13_9_10_B is an allowed parallel pipe'].

## 2. Cross-energy coupling consistency
- GFG mapping: gas junction -> power bus for gas-to-power operation; listed as bidirectional references below.
- GFG_0: power bus 18 <-> gas junction 4, Pmax=2.000 MW, eta=0.380
- GFG_1: power bus 22 <-> gas junction 5, Pmax=2.000 MW, eta=0.380
- GFG_2: power bus 32 <-> gas junction 18, Pmax=1.500 MW, eta=0.360
- P2G mapping: power bus -> gas junction for power-to-gas operation.
- P2G_0: power bus 7 <-> gas junction 7, Pmax=1.500 MW, eta=0.700
- P2G_1: power bus 24 <-> gas junction 14, Pmax=1.500 MW, eta=0.700
- P2G_2: power bus 30 <-> gas junction 19, Pmax=1.000 MW, eta=0.650
- Compressor mapping: electric load bus -> gas compressor arc.
- COMP_STATION_8_TO_9_EQ: electric bus 7 -> gas compressor arc 7->8 (fixed)
- COMP_17_TO_18: electric bus 30 -> gas compressor arc 16->17 (controllable)
- Conversion basis: MW = MJ/s and HHV = 50.000 MJ/kg.
- GFG formula: mdot_gas = P_e / (eta * HHV), giving kg/s gas consumption.
- P2G formula: mdot_gas = eta * P_e / HHV, giving kg/s gas injection.

## 3. Multi-time-scale consistency
- Fast step length: 3 minutes.
- Steps per day: 480 steps.
- Slow action interval: 20 fast steps (60 minutes).
- Manager interval: 40 fast steps.
- Training constants source: import.
- Action-space decomposition: slow=10, fast=16, total=26.
- Goal dimension, if available: 32.

## 4. Constraint modeling consistency
- Voltage limits: 0.950 to 1.050 p.u.
- Gas pressure limits: 2.500 to 5.000 bar; target 4.000 bar.
- Pipe velocity limit: 12.000 m/s.
- SOC limits across ESS devices: 0.100 to 0.950.
- Source capacity limits: {0: 0.55, 7: 0.2} kg/s by source node.

## 5. 24-hour simulation sanity check
- Baseline policy: mild_baseline; a deterministic mild rule policy with gentle ESS charge/discharge, small GFG/P2G fractions, initial compressor ratio, limited inverter Q support, and modest renewable curtailment.
- Steps recorded: 480.
- Power flow success rate: 1.000.
- Gas flow success rate: 1.000.
- Solver failure count: 0.
- Observed voltage range: 0.9584 to 1.0490 p.u.
- Observed gas pressure range: 3.9058 to 4.9394 bar.
- Observed SOC range: 0.5000 to 0.8680.
- Major constraint violation under this baseline: False.
- Metrics not read from current info/env fields: none.

## 6. Modeling scope and caveats
- Gas model name: Belgian-20-derived medium-pressure micro gas distribution network.
- The gas model is a Belgian-20-derived medium-pressure research calibration, not an exact engineering reproduction of the original Belgian transmission network.
- The gas model is event-triggered quasi-steady-state, not a full transient gas dynamics model.
- This scope is suitable for a reinforcement-learning scheduling testbed for electric-gas coupled microgrid research.
- Calibration notes from the simulator:
  - Belgian-20-derived medium-pressure micro gas distribution network (belgian20_derived_mp_v2).
  - Node names are Belgian-20 topology-source labels only; pipe length, diameter, pressure, load, source and compressor data are medium-pressure research calibrations.
  - The 7->8 station is a fixed-ratio hydraulic compressor with equivalent_units=2; flow and power values already represent the aggregated station.
  - Only COMP_17_TO_18 is exposed to the RL slow action in belgian20_derived_mp_v2.
  - P2G includes gas conditioning and injection boosting in the aggregate efficiency; no extra P2G compressors are modeled.
  - Supplier marginal costs, pipe roughness, gas composition and all device ratings remain research assumptions unless project data are supplied.

## Generated artifacts
- static_topology: `model_reasonableness_figures\01_static_topology_coupling.png`
- device_capacity_action_space: `model_reasonableness_figures\02_device_capacity_and_action_space.png`
- multiscale_timeline: `model_reasonableness_figures\03_multiscale_timeline.png`
- timeseries_csv: `model_reasonableness_figures\model_reasonableness_timeseries.csv`
- constraint_envelopes: `model_reasonableness_figures\04_constraint_envelopes_24h.png`
- coupling_flow_consistency: `model_reasonableness_figures\05_coupling_flow_consistency.png`
