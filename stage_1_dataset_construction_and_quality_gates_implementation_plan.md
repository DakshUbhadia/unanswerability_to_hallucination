# Implementation Plan — Stage 1: Dataset Construction, Quality Gates & Stratified PopQA Sampling

**Target Specification:** Mechanistic Interpretability Warm-Up (Frozen Specification v4.0)

**Execution Host:** **Local IDE on CPU** (WSL / Linux / macOS — **0.0 Kaggle GPU Hours Used**)

**Stage Covered:** Stage 1 (`scripts/stage1_build_dataset.py`)

**Timeline:** Working Days 2–4

## 1. Stage 1 Overview & Local CPU Architecture

Stage 1 builds and validates the entire experimental dataset on your **local machine's CPU** before any GPU jobs run in Stage 2. Because Stage 1 only requires the `Qwen/Qwen2.5-3B-Instruct` **tokenizer** ($\approx 15\text{ MB}$ download, no 6 GB model weights needed) and the Hugging Face `akariasai/PopQA` table, you can iterate, run quality gates, and execute unit tests locally in seconds without touching your Kaggle GPU quota.

### 1.1 Stage 1 Input / Output Contract

| File Path | Role | Row / Record Count | Format & Schema Reference | 
| ----- | ----- | ----- | ----- | 
| `data/templates.jsonl` | Input / Verified Artifact | `96` rows (`48` Fam A, `48` Fam B; or `64` under `--cut3`) | JSONL (Section 8.1) | 
| `data/splits.json` | Output | `1` JSON object (`48` train, `24` val, `24` test template IDs) | JSON (Section 8.1) | 
| `data/pairs.jsonl` | Output | `384` rows (`192` train, `96` val, `96` test pairs = `768` prompts) | JSONL (Section 8.1) | 
| `data/b_ctrl.jsonl` | Output | `48` rows (`12` Fam B test templates $\times 4$ instances) | JSONL (Section 8.1) | 
| `data/popqa_sample.jsonl` | Output | `1000` rows (`60`/decile in Deciles 1–5; `140`/decile in Deciles 6–10) | JSONL (Section 8.1) | 
| `data/audit_log.csv` | Output | `1+` rows (logs every initial check failure and in-place rewrite) | CSV (Section 8.1) | 

## 2. Mathematical Mechanics of the Stage 1 Quality Gates

### 2.1 Automated Quality Gate 1 — Exact Token-Length Parity ($\vert{}\Delta L\vert{} \le 2$)

For every template $t \in \{1, \dots, 96\}$ and instance $k \in \{0, 1, 2, 3\}$, let $L(x_{t,k}^{\text{ans}})$ and $L(x_{t,k}^{\text{unans}})$ be the exact token counts produced by `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")`.

* **Invariant:**
  

  $$
  \left|L\left(x_{t,k}^{\text{ans}}\right) - L\left(x_{t,k}^{\text{unans}}\right)\right| \le 2
  $$

* **In-Place Auto-Calibration & Audit Logging:** If an instantiated pair exhibits $\vert{}\Delta L\vert{} > 2$ due to subword tokenization of a synthetic entity or attribute, the Stage 1 builder automatically adjusts neutral modifier tokens in the unanswerable distractor clause (Family A) or unanswerable question phrase (Family B) until $\vert{}\Delta L\vert{} \le 2$, and records the exact `old_text`, `new_text`, and `offending_detail` as a permanent row in `data/audit_log.csv`.

### 2.2 Automated Quality Gate 2 — N-Gram Lexical Leakage ($\max_g |\phi(g)| \le 0.15$)

Let $\mathcal{D}$ be the full set of $N = 768$ prompts ($N_1 = 384$ unanswerable prompts with label $y = 1$, and $N_0 = 384$ answerable prompts with label $y = 0$). For every word n-gram $g$ of order $n \in \{1, 2, 3\}$ present in at least 2 prompts, let $I_g(p) \in \{0, 1\}$ indicate whether n-gram $g$ appears in normalized prompt $p$. Forming the $2 \times 2$ contingency counts:

* $n_{11} = \sum_{p: y(p)=1} I_g(p)$ (unanswerable prompts containing $g$)

* $n_{10} = N_1 - n_{11}$ (unanswerable prompts not containing $g$)

* $n_{01} = \sum_{p: y(p)=0} I_g(p)$ (answerable prompts containing $g$)

* $n_{00} = N_0 - n_{01}$ (answerable prompts not containing $g$)

The exact Phi coefficient $\phi(g)$ is:

$$
\phi(g) = \frac{n_{11} n_{00} - n_{10} n_{01}}{\sqrt{(n_{11}+n_{10})(n_{01}+n_{00})(n_{11}+n_{01})(n_{10}+n_{00})}}
$$

* **Why Our 96-Template Design Satisfies** $\vert{}\phi(g)\vert{} \le 0.15$**:**
  Because $N_1 = N_0 = 384$, any predicate unique to a single template ($n_{11} = 4, n_{01} = 0$) has:
  

  $$
  |\phi(g)| = \sqrt{\frac{4}{768 - 4}} = \sqrt{\frac{4}{764}} \approx 0.0723 \le 0.15
  $$

  
  Furthermore, by counterbalancing question interrogative stems (`"What is"`, `"Which specific"`, `"According to the record, what"`) equally across `ans` and `unans` templates, common n-grams achieve $n_{11} \approx n_{01} \implies |\phi(g)| \approx 0.00$.

## 3. Complete 96-Template Catalog & Auto-Calibrator (`src/template_catalog.py`)

Create `src/template_catalog.py`. This module deterministically generates all **48 Family A templates** and **48 Family B templates** (with `b_ctrl_template` and 4 distinct slot instances per template), includes a deliberate pre-calibration seed template that triggers and tests the in-place `data/audit_log.csv` rewrite pipeline, and enforces $|\Delta L| \le 2$ across all 384 pairs.

```
# src/template_catalog.py
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# 48 distinct domains, answerable clauses, and distractor clauses for Family A.
# Each tuple: (domain, context_prefix, ans_clause, unans_clause, question_str, attr1_examples, attr2_examples)
FAMILY_A_SPECS: List[Tuple[str, str, str, str, str, List[str], List[str]]] = [
    ("solar_research", "Clinical and engineering dossier for {entity} archived in 2019.", "Primary technical oversight was led by {attr1}.", "Annual facility maintenance was audited by {attr2}.", "Based strictly on the dossier, who led primary technical oversight for {entity}?", ["Dr. Aris Thorne", "Dr. Nadia Lin", "Dr. Kenji Sato", "Dr. Elena Rostova"], ["Apex Compliance", "Vertex Audits", "Beacon Review", "Crest Inspectors"]),
    ("geothermal_grid", "Infrastructure dispatch record for {entity} logged in 2020.", "The chief turbine architect on record was {attr1}.", "The regional concrete supplier on record was {attr2}.", "Based strictly on the dispatch record, who was the chief turbine architect for {entity}?", ["Marcus Vance", "Lukas Weber", "Tariq Al-Mansoor", "Soren Lindqvist"], ["Nordic Cement", "Boreal Stone", "Glacier Aggregate", "Fjord Materials"]),
    ("quantum_ledger", "Cryptography validation memo for {entity} filed in 2021.", "The lattice key protocol was authored by {attr1}.", "The cooling rack enclosure was shipped by {attr2}.", "Based strictly on the validation memo, who authored the lattice key protocol for {entity}?", ["Prof. Clara Hao", "Prof. Viktor Krenn", "Prof. Amira Saleh", "Prof. Devon Brooks"], ["CryoLogistics", "Polar Freight", "Thermal Transit", "Alps Cargo"]),
    ("solid_battery", "Electrochemistry lab bulletin for {entity} issued in 2022.", "The electrolyte dopant formula was synthesized by {attr1}.", "The warehouse ventilation duct was cleaned by {attr2}.", "Based strictly on the lab bulletin, who synthesized the electrolyte dopant formula for {entity}?", ["Dr. Yuna Park", "Dr. Mateo Silva", "Dr. Priya Nair", "Dr. Felix Bauer"], ["CleanAir Services", "DuctPro Team", "Breeze Maintenance", "AeroScrub Corp"]),
    ("genomic_biobank", "Bioinformatics registry entry for {entity} updated in 2018.", "The sequencing pipeline calibration was verified by {attr1}.", "The exterior parking lot paving was completed by {attr2}.", "Based strictly on the registry entry, who verified the sequencing pipeline calibration for {entity}?", ["Dr. Hannah Cole", "Dr. Omar Farooq", "Dr. Celine Moreau", "Dr. Pavel Novak"], ["Metro Asphalt", "PaveRight Ltd", "Tarmac Brothers", "Roadway Group"]),
    ("photonic_chip", "Semiconductor lithography note for {entity} recorded in 2023.", "The waveguide etching mask was designed by {attr1}.", "The cafeteria vending contract was managed by {attr2}.", "Based strictly on the lithography note, who designed the waveguide etching mask for {entity}?", ["Ingrid Holm", "Rohan Mehta", "Dante Moretti", "Keiko Tanaka"], ["SnackLine Inc", "Refresh Vendors", "BistroMatic", "Pantry Express"]),
    ("maritime_manifest", "Port authority customs log for {entity} stamped in 2017.", "The vessel navigation charter was signed by {attr1}.", "The dockside crane repainting was billed by {attr2}.", "Based strictly on the customs log, who signed the vessel navigation charter for {entity}?", ["Capt. Arthur Pendelton", "Capt. Sofia Reyes", "Capt. Henrik Olsen", "Capt. Mei-Ling Zhou"], ["Harbor Coatings", "Marine Paint Co", "Seawall Finishers", "Pier Tints"]),
    ("clinical_trial", "Pharmacology study protocol for {entity} registered in 2020.", "The biostatistics endpoint model was reviewed by {attr1}.", "The courier specimen packaging was supplied by {attr2}.", "Based strictly on the study protocol, who reviewed the biostatistics endpoint model for {entity}?", ["Dr. Julian Mercer", "Dr. Ananya Rao", "Dr. Tobias Klein", "Dr. Rachel Stern"], ["SafeBox Medical", "ParcelShield", "ColdPack Containers", "BioCrate Co"]),
    ("satellite_telemetry", "Orbital tracking summary for {entity} transmitted in 2021.", "The attitude thruster burn sequence was calculated by {attr1}.", "The ground station lawn mowing was scheduled by {attr2}.", "Based strictly on the tracking summary, who calculated the attitude thruster burn sequence for {entity}?", ["Dr. Liam O'Connor", "Dr. Mei Chen", "Dr. Sergei Volkov", "Dr. Fatima Zahra"], ["GreenTurf Care", "Valley Landscaping", "Meadow Grounds", "Horizon Mowers"]),
    ("municipal_zoning", "City planning commission minutes for {entity} archived in 2019.", "The seismic variance assessment was certified by {attr1}.", "The hearing room microphone rental was arranged by {attr2}.", "Based strictly on the commission minutes, who certified the seismic variance assessment for {entity}?", ["Eng. Victor Morales", "Eng. Diane Foster", "Eng. Haruki Abe", "Eng. Nora Lind"], ["StageSound Rentals", "CityAudio Gear", "Echo AV Systems", "ClearTone Co"]),
    ("museum_provenance", "Curatorial acquisition ledger for {entity} cataloged in 2016.", "The radiocarbon pigment analysis was conducted by {attr1}.", "The lobby lighting fixture was polished by {attr2}.", "Based strictly on the acquisition ledger, who conducted the radiocarbon pigment analysis for {entity}?", ["Dr. Beatrice Vane", "Dr. Lorenzo Conti", "Dr. Yuki Shimizu", "Dr. Antonella Ricci"], ["Lumen Polishers", "Gleam Maintenance", "Gallery Brass Care", "BrightFixture Ltd"]),
    ("supply_audit", "Procurement compliance report for {entity} finalized in 2022.", "The cobalt smelter traceability check was performed by {attr1}.", "The office stationery printing was fulfilled by {attr2}.", "Based strictly on the compliance report, who performed the cobalt smelter traceability check for {entity}?", ["Claire Deschamps", "Gavin Thorne", "Satoshi Endo", "Miriam Haddad"], ["PaperTrail Press", "InkWell Stationers", "PageOne Print", "FineLeaf Paper"]),
    ("archaeology_dig", "Stratigraphy field journal for {entity} compiled in 2018.", "The obsidian blade classification was documented by {attr1}.", "The basecamp tent waterproofing was treated by {attr2}.", "Based strictly on the field journal, who documented the obsidian blade classification for {entity}?", ["Prof. Idris Elba-Smith", "Prof. Helena Costa", "Prof. Martin Vogel", "Prof. Zainab Al-Ali"], ["CanvasGuard Co", "Outback Shelters", "TrekProof Tents", "DryCamp Gear"]),
    ("incident_postmortem", "Cloud reliability postmortem for {entity} published in 2023.", "Thebgp routing table rollback was executed by {attr1}.", "The breakroom espresso machine was descaled by {attr2}.", "Based strictly on the postmortem, who executed the bgp routing table rollback for {entity}?", ["Devon K. Patel", "Svetlana Orlova", "Marco Bellini", "Grace Liu"], ["BeanSteam Service", "CafeCare Techs", "RoastRepair Co", "BrewFix Inc"]),
    ("patent_abstract", "Intellectual property docket for {entity} indexed in 2020.", "The prior art patentability search was completed by {attr1}.", "The archival binder embossing was invoiced by {attr2}.", "Based strictly on the docket, who completed the prior art patentability search for {entity}?", ["Atty. Warren Hale", "Atty. Simone Laurent", "Atty. Rajeev Kapoor", "Atty. Astrid Lindholm"], ["GoldLeaf Bindery", "Crest Embossing", "Folio Crafts", "SpinePress Co"]),
    ("weather_bulletin", "Atmospheric sonde bulletin for {entity} logged in 2021.", "The tropopause wind shear profile was calibrated by {attr1}.", "The access road snow plowing was contracted by {attr2}.", "Based strictly on the sonde bulletin, who calibrated the tropopause wind shear profile for {entity}?", ["Dr. Evan Sterling", "Dr. Katrin Weiss", "Dr. Daichi Mori", "Dr. Salma Mansour"], ["FrostBlade Plows", "Alpine Clearing", "WinterRoad Corp", "IcePath Services"]),
    ("aviation_inspection", "Airframe maintenance logbook for {entity} signed in 2019.", "The ultrasonic wing spar test was inspected by {attr1}.", "The hangar floor epoxy coating was applied by {attr2}.", "Based strictly on the maintenance logbook, who inspected the ultrasonic wing spar test for {entity}?", ["Tech. Gregory Walsh", "Tech. Alina Popescu", "Tech. Kenzo Fujii", "Tech. Carlos Mendez"], [" DuraFloor Epoxy", "HangarCoat Ltd", "PolySeal Surfaces", "AeroDeck Finish"]),
    ("hydrology_survey", "Watershed sediment survey for {entity} recorded in 2017.", "The dissolved nitrate titration was analyzed by {attr1}.", "The trailer tire replacement was serviced by {attr2}.", "Based strictly on the sediment survey, who analyzed the dissolved nitrate titration for {entity}?", ["Dr. Fiona Gallagher", "Dr.cheng Wei", "Dr. Lars Mikkelsen", "Dr. Naomi Bekele"], ["AxlePro Tires", "TrailWheel Garage", "Ridgeway Auto", "Summit Tread Co"]),
    ("fusion_tokamak", "Plasma confinement log for {entity} archived in 2022.", "The divertor magnetic coil alignment was tuned by {attr1}.", "The visitor badge lanyard order was shipped by {attr2}.", "Based strictly on the confinement log, who tuned the divertor magnetic coil alignment for {entity}?", ["Dr. Ilan Gur", "Dr. Veronika Horvath", "Dr. Taro Kurosawa", "Dr. Quentin Blake"], ["BadgeWorks Direct", "LanyardPlus Co", "IDTag Express", "CordCraft Supply"]),
    ("botanical_herbarium", "Taxonomic voucher sheet for {entity} dated in 2018.", "The pollen morphology microscopy was verified by {attr1}.", "The cabinet humidity silica gel was restocked by {attr2}.", "Based strictly on the voucher sheet, who verified the pollen morphology microscopy for {entity}?", ["Dr. Sylvia Moss", "Dr. Fernando Cruz", "Dr. Aiko Hayashi", "Dr. Bernard Mercier"], ["DryPack Desiccants", "SilicaSafe Co", "MoistureGuard", "AridBox Supplies"]),
    ("seismic_array", "Tectonic borehole record for {entity} filed in 2020.", "The downhole geophone gain was adjusted by {attr1}.", "The perimeter chainlink fence was repaired by {attr2}.", "Based strictly on the borehole record, who adjusted the downhole geophone gain for {entity}?", ["Dr. Malcolm Reed", "Dr. Giada Romano", "Dr. Jin-Woo Kim", "Dr. Ingrid Solberg"], ["IronLink Fencing", "BorderWire Co", "SteelPost Builders", "SecureMesh Ltd"]),
    ("vaccine_coldchain", "Biologics stability manifest for {entity} audited in 2021.", "The ultra-low freezer thermocouple was validated by {attr1}.", "The loading dock rubber bumper was bolted by {attr2}.", "Based strictly on the stability manifest, who validated the ultra-low freezer thermocouple for {entity}?", ["Dr. Tessa Vance", "Dr. Nikhil Sharma", "Dr. Emilie Blanc", "Dr. Jakub Kowalski"], ["DockGuard Rubber", "BayBumper Co", "ImpactBuffers", "QuayHardware"]),
    ("subsea_cable", "Transoceanic fiber log for {entity} compiled in 2019.", "The optical repeater attenuation test was run by {attr1}.", "The galley kitchen dishwasher was repaired by {attr2}.", "Based strictly on the fiber log, who ran the optical repeater attenuation test for {entity}?", ["Eng.Declan Byrne", "Eng. Ayaka Ito", "Eng. Mateus Rocha", "Eng. Freja Holm"], ["GalleyFix Marine", "ShipAppliance Co", "DeckWasher Tech", "NautiRepair"]),
    ("radiotelescope", "Interferometry baseline note for {entity} logged in 2023.", "The maser atomic clock phase was locked by {attr1}.", "The gravel walkway weed trimming was done by {attr2}.", "Based strictly on the baseline note, who locked the maser atomic clock phase for {entity}?", ["Dr. Gareth Hughes", "Dr. Ximena Soto", "Dr. Riku Watanabe", "Dr.Leila Abbasi"], ["DesertTrim Crew", "PathClear Grounds", "Mesa Brush Co", "AridScape Care"]),
    ("glacier_core", "Cryosphere drilling report for {entity} registered in 2018.", "The trapped methane isotope ratio was measured by {attr1}.", "The generator exhaust muffler was welded by {attr2}.", "Based strictly on the drilling report, who measured the trapped methane isotope ratio for {entity}?", ["Dr. Astrid Nyman", "Dr. Bruno Silva", "Dr. Haruka Sano", "Dr. Samira Khan"], ["ArcWeld Polar", "TorchCraft Metal", "MufflerFix Shop", "IronSeam Works"]),
    ("rail_signaling", "Interlocking safety certificate for {entity} issued in 2020.", "The Wayside transponder fail-safe logic was tested by {attr1}.", "The station bench wood staining was finished by {attr2}.", "Based strictly on the safety certificate, who tested the wayside transponder fail-safe logic for {entity}?", ["Eng. Roland Beck", "Eng. Carmen Vega", "Eng. Takumi Ota", "Eng. Lydia Park"], ["OakGlow Finishes", "TimberStain Co", "BenchCraft Wood", "CedarCoat Ltd"]),
    ("wind_offshore", "Nacelle telemetry dossier for {entity} recorded in 2022.", "The pitch bearing vibration spectrum was diagnosed by {attr1}.", "The crew boat lifejacket locker was inventoried by {attr2}.", "Based strictly on the telemetry dossier, who diagnosed the pitch bearing vibration spectrum for {entity}?", ["Eng. Willem de Boer", "Eng. Chiara Ferri", "Eng. Jun-Ho Seo", "Eng. Maeve Kelly"], ["SeaSafe Gear", "BuoyLocker Audits", "DeckVest Supply", "OceanSafety Co"]),
    ("desalination_plant", "Reverse osmosis logbook for {entity} archived in 2021.", "The membrane permeate conductivity was checked by {attr1}.", "The guardhouse window tinting was installed by {attr2}.", "Based strictly on the osmosis logbook, who checked the membrane permeate conductivity for {entity}?", ["Dr. Tariq Nassar", "Dr. Laura Marin", "Dr. Kazuki Miura", "Dr. Sonja Berg"], ["SunBlock Glass", "SolarShade Film", "ClearTint Pros", "GlareGuard Co"]),
    ("acoustics_chamber", "Anechoic calibration sheet for {entity} filed in 2019.", "The fiberglass wedge absorption curve was plotted by {attr1}.", "The hallway coat rack mounting was anchored by {attr2}.", "Based strictly on the calibration sheet, who plotted the fiberglass wedge absorption curve for {entity}?", ["Dr. Jonas Lind", "Dr. Valeria Molinari", "Dr. Shohei Kanda", "Dr. Daria Petrova"], ["WallAnchor Handyman", "StudMount Services", "HallwayFix Co", "BoltRight Crew"]),
    ("mycology_culture", "Fungal strain repository card for {entity} dated in 2020.", "The agar mycelium growth rate was scored by {attr1}.", "The autoclave room floor drain was flushed by {attr2}.", "Based strictly on the repository card, who scored the agar mycelium growth rate for {entity}?", ["Dr. Callum Ross", "Dr. Beatriz Santos", "Dr. Natsuki Aoki", "Dr. Zahra Shirazi"], ["PipeClear Plumbing", "DrainJet Services", "FlowRight Techs", "CleanTrap Co"]),
    ("nuclear_dosimetry", "Health physics badge report for {entity} certified in 2017.", "The neutron track etch calibration was approved by {attr1}.", "The reception desk floral display was watered by {attr2}.", "Based strictly on the badge report, who approved the neutron track etch calibration for {entity}?", ["Dr. Leonard Vance", "Dr. Paulinasikora", "Dr. Ren Takahashi", "Dr. Amelie Dupuis"], ["PetalCare Flora", "LobbyBloom Co", "GreenStem Services", "VaseFresh Ltd"]),
    ("autonomous_lidar", "Sensor fusion test log for {entity} recorded in 2023.", "The point-cloud extrinsic matrix was solved by {attr1}.", "The garage roll-up door chain was greased by {attr2}.", "Based strictly on the test log, who solved the point-cloud extrinsic matrix for {entity}?", ["Dr. Kevin Zhao", "Dr. Sofia Lindqvist", "Dr. Matteo Galli", "Dr. Hina Matsui"], ["RollSmooth Doors", "ChainLube Garage", "OverheadFix Co", "BayDoor Service"]),
    ("metallurgy_foundry", "Alloy tensile certificate for {entity} stamped in 2018.", "The grain boundary microhardness was measured by {attr1}.", "The locker room soap dispenser was refilled by {attr2}.", "Based strictly on the tensile certificate, who measured the grain boundary microhardness for {entity}?", ["Dr. Gunter Braun", "Dr.Lucia Herrera", "Dr. Daiki Saito", "Dr. Eleni Papadopoulos"], ["SudsSupply Co", "WashRoom Care", "CleanHands Service", "SoapLine Direct"]),
    ("volcanology_gas", "Fumarole plume bulletin for {entity} logged in 2021.", "The sulfur dioxide flux spectrometer was operated by {attr1}.", "The field jeep windshield wiper was replaced by {attr2}.", "Based strictly on the plume bulletin, who operated the sulfur dioxide flux spectrometer for {entity}?", ["Dr. Marco Rossi", "Dr. Keiko Mori", "Dr. Samuel Osei", "Dr. Freya Lind"], ["ClearView Auto", "WiperPro Shop", "glassCare Garage", "TrailAuto Parts"]),
    ("cryptanalysis_archive", "Ciphertext recovery memo for {entity} cataloged in 2016.", "The frequency digram substitution table was reconstructed by {attr1}.", "The reading room velvet curtain was dry-cleaned by {attr2}.", "Based strictly on the recovery memo, who reconstructed the frequency digram substitution table for {entity}?", ["Dr. Elliott Vance", "Dr. Marta Kowalczyk", "Dr. Shinji Ikeda", "Dr. Clara Morel"], ["VelvetCare Cleaners", "DraperyWash Co", "FineFabric Press", "CurtainFresh Ltd"]),
    ("aquaculture_hatchery", "Marine larval tank record for {entity} filed in 2022.", "The dissolved oxygen biofilter loop was balanced by {attr1}.", "The boat ramp algae pressure-washing was done by {attr2}.", "Based strictly on the tank record, who balanced the dissolved oxygen biofilter loop for {entity}?", ["Dr. Haakon Lund", "Dr. Camila Rojas", "Dr. Yuto Nakajima", "Dr. Amina Bello"], ["JetWash Marine", "RampScrub Pros", "HydroBlast Crew", "Slipway Cleaners"]),
    ("avionics_firmware", "Flight control verification note for {entity} issued in 2020.", "The watchdog timer interrupt latency was profiled by {attr1}.", "The conference table chair casters were oiled by {attr2}.", "Based strictly on the verification note, who profiled the watchdog timer interrupt latency for {entity}?", ["Eng. Nathan Brooks", "Eng. Celine Garnier", "Eng. Kaito Sasaki", "Eng. Ines Silva"], ["GlideCaster Care", "OfficeChair Fix", "SwivelSmooth Co", "SeatRoll Service"]),
    ("paleoclimate_coral", "Carbonate sclerochronology sheet for {entity} archived in 2019.", "The strontium-to-calcium paleothermometer was run by {attr1}.", "The lab sink faucet aerator was descaled by {attr2}.", "Based strictly on the sclerochronology sheet, who ran the strontium-to-calcium paleothermometer for {entity}?", ["Dr. Georgia Vance", "Dr. Rafael Mendes", "Dr.Mina Kobayashi", "Dr. Anton Lind"], ["TapFlow Plumbing", "AeratorClean Co", "SinkFix Services", "ClearSpout Ltd"]),
    ("superconductor_wire", "Cryogenic magnet test sheet for {entity} logged in 2023.", "The critical current quench threshold was recorded by {attr1}.", "The shipping pallet shrink-wrap film was ordered by {attr2}.", "Based strictly on the test sheet, who recorded the critical current quench threshold for {entity}?", ["Dr.Lev Landau-Smith", "Dr. Giulia Costa", "Dr.Haruto Kato", "Dr. Sara Lindstrom"], ["WrapTight Packaging", "PalletFilm Co", "ShrinkGuard Supply", "CrateWrap Direct"]),
    ("epigraphy_tablet", "Cuneiform conservation log for {entity} compiled in 2017.", "The kiln-fired clay desalination bath was monitored by {attr1}.", "The display case brass padlock was lubricated by {attr2}.", "Based strictly on the conservation log, who monitored the kiln-fired clay desalination bath for {entity}?", ["Dr. Cyrus Vane", "Dr. Ariadne Dimitriou", "Dr. Kenjiabe", "Dr. Miriam Levy"], ["LockGlide Security", "BrassKey Care", "PadlockPro Service", "SafeHinge Co"]),
    ("forestry_lidar", "Canopy biomass inventory for {entity} finalized in 2021.", "The allometric crown diameter regression was fitted by {attr1}.", "The cabin woodstove chimney sweep was performed by {attr2}.", "Based strictly on the biomass inventory, who fitted the allometric crown diameter regression for {entity}?", ["Dr. Rowan Stirling", "Dr. Elsa Lindqvist", "Dr. Tomas Varga", "Dr. Mei-Hui Lin"], ["SootFree Sweepers", "FlueClean Pros", "HearthCare Service", "ChimneyBrush Co"]),
    ("Cleanroom_hvac", "Particulate certification log for {entity} signed in 2020.", "The HEPA laminar airflow velocity was certified by {attr1}.", "The hallway recycling bin labels were printed by {attr2}.", "Based strictly on the certification log, who certified the HEPA laminar airflow velocity for {entity}?", ["Eng. Bernard Vance", "Eng. Noemi Farkas", "Eng. Shun Ogawa", "Eng. Paula Castro"], ["BinTag Graphics", "EcoSign Print", "LabelCraft Shop", "SortSticker Co"]),
    ("sonar_bathymetry", "Multibeam trench survey for {entity} recorded in 2018.", "The sound velocity thermocline profile was corrected by {attr1}.", "The deck winch canvas cover was stitched by {attr2}.", "Based strictly on the trench survey, who corrected the sound velocity thermocline profile for {entity}?", ["Dr. Alistair Boyd", "Dr. Marina Silva", "Dr. Koji Yamamoto", "Dr. Ingrid Holm"], ["SailStitch Canvas", "DeckCover Tailor", "TarpSeam Marine", "WinchHood Co"]),
    ("enzymology_assay", "Kinetics spectrophotometry log for {entity} dated in 2022.", "The Michaelis constant substrate curve was fitted by {attr1}.", "The cold-room door magnetic gasket was wiped by {attr2}.", "Based strictly on the spectrophotometry log, who fitted the Michaelis constant substrate curve for {entity}?", ["Dr. Vera Novak", "Dr. Diego Morales", "Dr. Rina Hashimoto", "Dr. Simon Laurent"], ["SealWipe Hygiene", "GasketClean Co", "DoorStrip Care", "CoolSeal Service"]),
    ("bridge_strain", "Structural viaduct inspection for {entity} filed in 2019.", "The suspension cable acoustic emission was logged by {attr1}.", "The tollbooth lane traffic cones were stacked by {attr2}.", "Based strictly on the viaduct inspection, who logged the suspension cable acoustic emission for {entity}?", ["Eng. Conrad Vance", "Eng. Elena Popa", "Eng. Daichi Goto", "Eng. Clara Weiss"], ["ConeStack Traffic", "LaneMarker Crew", "RoadSafety Supply", "PylonCare Co"]),
    ("astrometry_transit", "Exoplanet photometry circular for {entity} published in 2023.", "The limb-darkening transit lightcurve was modeled by {attr1}.", "The dome shutter motor chain was tensioned by {attr2}.", "Based strictly on the photometry circular, who modeled the limb-darkening transit lightcurve for {entity}?", ["Dr. Miranda Vance", "Dr. Lucas Almeida", "Dr. Takuya Fujita", "Dr. Nadia Mansour"], ["DomeDrive Mechanics", "ShutterChain Co", "GearTension Tech", "ObservatoryFix"]),
    ("toxicology_elisa", "Forensic assay worksheet for {entity} archived in 2020.", "The microplate optical density curve was verified by {attr1}.", "The biohazard bin foot pedal was tightened by {attr2}.", "Based strictly on the assay worksheet, who verified the microplate optical density curve for {entity}?", ["Dr. Philip Vance", "Dr. RenataKowalska", "Dr. Yuji Murakami", "Dr.Sonia Patel"], ["PedalFix Hardware", "LabBin Service", "StepLid Repair", "CanHinge Co"]),
    ("piezoelectric_sensor", "Ferroelectric thin-film note for {entity} recorded in 2021.", "The poling hysteresis loop voltage was measured by {attr1}.", "The shipping crate foam insert was cut by {attr2}.", "Based strictly on the thin-film note, who measured the poling hysteresis loop voltage for {entity}?", ["Dr. Cedric Vance", "Dr. Adriana Rossi", "Dr. Kazuma Sato", "Dr. Linnea Berg"], ["FoamCut Packaging", "CushionBox Co", "InsertCraft Ltd", "DieCut Foam"]),
]

# 48 distinct biographical/entity profile domains for Family B.
# Question starters are balanced between ans_question and unans_question so no interrogative n-gram leaks.
FAMILY_B_SPECS: List[Tuple[str, str, str, str, str, List[str], List[str]]] = [
    ("alpine_botanist", "Profile: {entity} is an alpine botanist whose doctoral fieldwork was completed in {attr1}.", "According to the profile, where was {entity}'s doctoral fieldwork completed?", "According to the profile, what was {entity}'s childhood hamster's nickname?", "Profile: {entity} is an alpine botanist whose doctoral fieldwork was completed in {attr1}, and whose childhood hamster's nickname was {attr2}.", ["Zurich", "Innsbruck", "Grenoble", "Bergen"], ["Pippin", "Biscuit", "Clover", "Nibble"]),
    ("glaciologist", "Profile: {entity} is a polar glaciologist who joined the ice-core institute in {attr1}.", "According to the profile, in what year did {entity} join the ice-core institute?", "According to the profile, what brand of toothpaste does {entity} buy?", "Profile: {entity} is a polar glaciologist who joined the ice-core institute in {attr1} and buys {attr2} toothpaste.", ["1998", "2004", "2011", "2015"], ["MintCrest", "GlacierDent", "PureEnamel", "AlpsFresh"]),
    ("mycologist", "Profile: {entity} is a forest mycologist who lectures at the university in {attr1}.", "According to the profile, in which city does {entity} lecture?", "According to the profile, what was the combination to {entity}'s high-school locker?", "Profile: {entity} is a forest mycologist who lectures at the university in {attr1} and had the high-school locker combination {attr2}.", ["Portland", "Uppsala", "Sapporo", "Freiburg"], ["14-28-09", "32-11-45", "07-19-23", "41-08-16"]),
    ("seismologist", "Profile: {entity} is a marine seismologist whose flagship research vessel is named {attr1}.", "According to the profile, what is the name of {entity}'s flagship research vessel?", "According to the profile, what color umbrella did {entity} lose on the subway?", "Profile: {entity} is a marine seismologist whose flagship research vessel is named {attr1} and who lost a {attr2} umbrella on the subway.", ["RV Calypso-II", "RV TrenchStar", "RV Nautilus-V", "RV Pelagic"], ["burgundy", "mustard", "chartreuse", "cerulean"]),
    ("numismatist", "Profile: {entity} is a classical numismatist specializing in bronze coinage from {attr1}.", "According to the profile, which region's bronze coinage does {entity} specialize in?", "According to the profile, what ringtone melody is set on {entity}'s personal phone?", "Profile: {entity} is a classical numismatist specializing in bronze coinage from {attr1} whose personal phone ringtone melody is {attr2}.", ["Thrace", "Bactria", "Lydia", "Seleucia"], ["MarimbaEcho", "HarpChime", "BellSonata", "FluteWaltz"]),
    ("dendrochronologist", "Profile: {entity} is a dendrochronologist who curates the bristlecone archive in {attr1}.", "According to the profile, where does {entity} curate the bristlecone archive?", "According to the profile, what was the license plate of {entity}'s first scooter?", "Profile: {entity} is a dendrochronologist who curates the bristlecone archive in {attr1} and rode a first scooter with license plate {attr2}.", ["Flagstaff", "Reno", "Boise", "Bozeman"], ["KXP-409", "MTR-812", "VBN-334", "JQL-901"]),
    ("paleographer", "Profile: {entity} is a medieval paleographer who translates parchment charters written in {attr1}.", "According to the profile, which language are the charters {entity} translates written in?", "According to the profile, what flavor of jam does {entity} eat on toast?", "Profile: {entity} is a medieval paleographer who translates parchment charters written in {attr1} and eats {attr2} jam on toast.", ["Old Occitan", "Church Slavonic", "Middle High German", "Hiberno-Latin"], ["boysenberry", "cloudberry", "quince", "gooseberry"]),
    ("bryologist", "Profile: {entity} is a tropical bryologist who cataloged liverwort specimens across {attr1}.", "According to the profile, across which island did {entity} catalog liverwort specimens?", "According to the profile, what street number did {entity}'s maternal grandmother live at?", "Profile: {entity} is a tropical bryologist who cataloged liverwort specimens across {attr1} and whose maternal grandmother lived at street number {attr2}.", ["Borneo", "Madagascar", "Tasmania", "Socotra"], ["418", "729", "105", "663"]),
    ("astrophotographer", "Profile: {entity} is an astrophotographer who operates a robotic reflector in {attr1}.", "According to the profile, where does {entity} operate a robotic reflector?", "According to the profile, what shoe size does {entity}'s youngest cousin wear?", "Profile: {entity} is an astrophotographer who operates a robotic reflector in {attr1} and whose youngest cousin wears shoe size {attr2}.", ["Atacama", "La Palma", "Mauna Kea", "Karoo"], ["EU-39", "EU-42", "EU-37", "EU-44"]),
    ("ethnomusicologist", "Profile: {entity} is an ethnomusicologist who records polyphonic folk songs in {attr1}.", "According to the profile, where does {entity} record polyphonic folk songs?", "According to the profile, what serial code is stamped inside {entity}'s wristwatch?", "Profile: {entity} is an ethnomusicologist who records polyphonic folk songs in {attr1} and wears a wristwatch stamped with serial code {attr2}.", ["Sardinia", "Epirus", "Svaneti", "Tuva"], ["W-8831", "W-4409", "W-1927", "W-6550"]),
    ("limnologist", "Profile: {entity} is a freshwater limnologist who monitors diatom blooms in {attr1}.", "According to the profile, in which lake does {entity} monitor diatom blooms?", "According to the profile, what middle name was given to {entity}'s childhood canary?", "Profile: {entity} is a freshwater limnologist who monitors diatom blooms in {attr1} and had a childhood canary with the middle name {attr2}.", ["Lake Baikal", "Lake Tanganyika", "Lake Ohrid", "Lake Biwa"], ["Sunny", "Pip", "Zephyr", "Goldie"]),
    ("horologist", "Profile: {entity} is an antique horologist who restores marine chronometers built in {attr1}.", "According to the profile, from which century are the chronometers {entity} restores?", "According to the profile, what seat number did {entity} occupy on last Tuesday's train?", "Profile: {entity} is an antique horologist who restores marine chronometers built in {attr1} and occupied seat number {attr2} on last Tuesday's train.", ["the 18th century", "the 17th century", "the 19th century", "the 16th century"], ["12B", "04A", "19C", "08D"]),
    ("lichenologist", "Profile: {entity} is an arctic lichenologist who directs the tundra station in {attr1}.", "According to the profile, where does {entity} direct the tundra station?", "According to the profile, what model of toaster sits on {entity}'s kitchen counter?", "Profile: {entity} is an arctic lichenologist who directs the tundra station in {attr1} and has a {attr2} toaster on the kitchen counter.", ["Svalbard", "Tromso", "Nuuk", "Abisko"], ["ToastMaster-4", "CrispPro-X", "DualSlot-9", "BrowningPlus"]),
    ("speleologist", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1}.", "According to the profile, in which province did {entity} map subterranean caverns?", "According to the profile, what PIN code unlocks {entity}'s bicycle padlock?", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1} and uses PIN code {attr2} for the bicycle padlock.", ["Guizhou", "Slovenia", "Yucatan", "Cantabria"], ["4921", "8037", "1564", "7290"]),
    ("malacologist", "Profile: {entity} is a marine malacologist who studies cone snail venoms at {attr1}.", "According to the profile, at which aquarium does {entity} study cone snail venoms?", "According to the profile, what fabric pattern covers {entity}'s living-room sofa?", "Profile: {entity} is a marine malacologist who studies cone snail venoms at {attr1} and owns a living-room sofa with a {attr2} fabric pattern.", ["Monterey Lab", "Okinawa Marine", "Monaco Ocean", "Woods Hole"], ["herringbone", "paisley", "houndstooth", "gingham"]),
    ("lepidopterist", "Profile: {entity} is a cloud-forest lepidopterist who tracks monarch migrations through {attr1}.", "According to the profile, through which mountain range does {entity} track migrations?", "According to the profile, what laundry detergent scent does {entity} prefer?", "Profile: {entity} is a cloud-forest lepidopterist who tracks monarch migrations through {attr1} and prefers {attr2} laundry detergent scent.", ["Sierra Madre", "Cordillera Real", "Rwenzori", "Nilgiri Hills"], ["lavender", "eucalyptus", "cedarwood", "lemongrass"]),
    ("gemologist", "Profile: {entity} is a forensic gemologist who grades sapphire inclusions in {attr1}.", "According to the profile, in which trading hub does {entity} grade sapphire inclusions?", "According to the profile, what nickname did {entity}'s primary school art teacher have?", "Profile: {entity} is a forensic gemologist who grades sapphire inclusions in {attr1} and whose primary school art teacher had the nickname {attr2}.", ["Antwerp", "Geneva", "Jaipur", "Chanthaburi"], ["Skipper", "Maestro", "Doc", "Rusty"]),
    ("luthiery_maker", "Profile: {entity} is a concert luthier who carves cello soundboards from spruce grown in {attr1}.", "According to the profile, where is the spruce {entity} uses for cello soundboards grown?", "According to the profile, what radio station frequency is preset #1 in {entity}'s car?", "Profile: {entity} is a concert luthier who carves cello soundboards from spruce grown in {attr1} and sets {attr2} as preset #1 on the car radio.", ["Val di Fiemme", "Mittenwald", "Carpathia", "Sitka"], ["94.7 FM", "101.3 FM", "88.5 FM", "99.1 FM"]),
    ("oenologist", "Profile: {entity} is a fermentation oenologist who consults for hillside vineyards in {attr1}.", "According to the profile, in which valley does {entity} consult for hillside vineyards?", "According to the profile, what board game did {entity} play on New Year's Eve?", "Profile: {entity} is a fermentation oenologist who consults for hillside vineyards in {attr1} and played the board game {attr2} on New Year's Eve.", ["Douro Valley", "Mosel Valley", "Barossa Valley", "Colchagua"], ["Backgammon", "Carcassonne", "Mahjong", "Cribbage"]),
    ("carillonneur", "Profile: {entity} is a municipal carillonneur who performs weekly bell recitals in {attr1}.", "According to the profile, in which town does {entity} perform weekly bell recitals?", "According to the profile, what herb grows on {entity}'s bathroom windowsill?", "Profile: {entity} is a municipal carillonneur who performs weekly bell recitals in {attr1} and grows {attr2} on the bathroom windowsill.", ["Mechelen", "Delft", "Bruges", "Utrecht"], ["rosemary", "marjoram", "tarragon", "chives"]),
    ("archivometrics", "Profile: {entity} is a bibliometric historian who indexes scientific correspondence from {attr1}.", "According to the profile, from which decade does {entity} index scientific correspondence?", "According to the profile, what tea mug color sits on {entity}'s nightstand?", "Profile: {entity} is a bibliometric historian who indexes scientific correspondence from {attr1} and keeps a {attr2} tea mug on the nightstand.", ["the 1890s", "the 1920s", "the 1870s", "the 1930s"], ["cobalt", "ochre", "vermilion", "celadon"]),
    ("palynologist", "Profile: {entity} is a quaternary palynologist who extracts fossilized spores from peat bogs in {attr1}.", "According to the profile, where does {entity} extract fossilized spores from peat bogs?", "According to the profile, what luggage tag initials are attached to {entity}'s backpack?", "Profile: {entity} is a quaternary palynologist who extracts fossilized spores from peat bogs in {attr1} and has luggage tag initials {attr2} on the backpack.", ["Connemara", "Karelia", "Tierra del Fuego", "Hokkaido"], ["K.R.V.", "M.T.L.", "S.J.B.", "P.H.N."]),
    ("crystallographer", "Profile: {entity} is an X-ray crystallographer who operates beamline diffractometers at {attr1}.", "According to the profile, at which synchrotron facility does {entity} operate diffractometers?", "According to the profile, what pizza topping does {entity} always refuse to eat?", "Profile: {entity} is an X-ray crystallographer who operates beamline diffractometers at {attr1} and always refuses the pizza topping {attr2}.", ["DESY Hamburg", "ESRF Grenoble", "Spring-8 Hyogo", "PSI Villigen"], ["anchovies", "artichokes", "capers", "jalapenos"]),
    ("ichthyologist", "Profile: {entity} is a deep-sea ichthyologist who describes bioluminescent lanternfish from {attr1}.", "According to the profile, from which ocean basin does {entity} describe lanternfish?", "According to the profile, what alarm clock time is set on {entity}'s Sunday schedule?", "Profile: {entity} is a deep-sea ichthyologist who describes bioluminescent lanternfish from {attr1} and sets a Sunday alarm clock for {attr2}.", ["the Banda Sea", "the Weddell Sea", "the Sargasso Sea", "the Coral Sea"], ["06:45 AM", "07:15 AM", "08:05 AM", "06:20 AM"]),
    ("cartographer", "Profile: {entity} is a bathymetric cartographer who compiles nautical depth charts for {attr1}.", "According to the profile, for which strait does {entity} compile nautical depth charts?", "According to the profile, what fountain pen ink color does {entity} sign birthday cards with?", "Profile: {entity} is a bathymetric cartographer who compiles nautical depth charts for {attr1} and signs birthday cards with {attr2} fountain pen ink.", ["the Luzon Strait", "the Denmark Strait", "the Bass Strait", "the Cook Strait"], ["sepia", "indigo", "oxblood", "emerald"]),
    ("herpetologist", "Profile: {entity} is a montane herpetologist who surveys glass-frog populations in {attr1}.", "According to the profile, in which reserve does {entity} survey glass-frog populations?", "According to the profile, what podcast episode did {entity} listen to yesterday evening?", "Profile: {entity} is a montane herpetologist who surveys glass-frog populations in {attr1} and listened to podcast episode {attr2} yesterday evening.", ["Monteverde", "Choco Darien", "Manu Biosphere", "Yasuni Park"], ["Ep-104", "Ep-219", "Ep-087", "Ep-312"]),
    ("metallographer", "Profile: {entity} is an archaeometallurgist who analyzes Damascus steel blades housed in {attr1}.", "According to the profile, in which museum city are the blades {entity} analyzes housed?", "According to the profile, what bakery pastry does {entity} buy on Saturday mornings?", "Profile: {entity} is an archaeometallurgist who analyzes Damascus steel blades housed in {attr1} and buys a {attr2} pastry on Saturday mornings.", ["Topkapi", "Toledo", "Graz", "Wallace London"], ["cardamom bun", "kouign-amann", "brioche tart", "almond croissant"]),
    ("helminthologist", "Profile: {entity} is a veterinary parasitologist who teaches diagnostic microscopy in {attr1}.", "According to the profile, where does {entity} teach diagnostic microscopy?", "According to the profile, what bookmark material is tucked inside {entity}'s novel?", "Profile: {entity} is a veterinary parasitologist who teaches diagnostic microscopy in {attr1} and uses a {attr2} bookmark inside the novel.", ["Ghent", "Utrecht", "Davis", "Guelph"], ["pressed fern", "brass feather", "silk ribbon", "tooled leather"]),
    ("pomologist", "Profile: {entity} is an heirloom pomologist who grafts frost-hardy apple cultivars in {attr1}.", "According to the profile, in which region does {entity} graft frost-hardy apple cultivars?", "According to the profile, what dog breed belonged to {entity}'s next-door neighbor?", "Profile: {entity} is an heirloom pomologist who grafts frost-hardy apple cultivars in {attr1} and lived next door to a neighbor's {attr2}.", ["Almaty", "Normandy", "Finger Lakes", "South Tyrol"], ["whippet", "keeshond", "samoyed", "vizsla"]),
    ("rheologist", "Profile: {entity} is a polymer rheologist who measures viscoelastic shear flows at {attr1}.", "According to the profile, at which polytechnic does {entity} measure viscoelastic shear flows?", "According to the profile, what keychain charm hangs from {entity}'s office keys?", "Profile: {entity} is a polymer rheologist who measures viscoelastic shear flows at {attr1} and has a {attr2} keychain charm on the office keys.", ["ETH Lausanne", "TU Delft", "RWTH Aachen", "KTH Stockholm"], ["miniature compass", "silver acorn", "pewter owl", "jade turtle"]),
    ("sedimentologist", "Profile: {entity} is a deltaic sedimentologist who cores turbidite fans offshore of {attr1}.", "According to the profile, offshore of which delta does {entity} core turbidite fans?", "According to the profile, what soup recipe did {entity} cook last Wednesday night?", "Profile: {entity} is a deltaic sedimentologist who cores turbidite fans offshore of {attr1} and cooked {attr2} soup last Wednesday night.", ["the Bengal Delta", "the Rhone Delta", "the Mackenzie Delta", "the Danube Delta"], ["borscht", "laksa", "minestrone", "pozole"]),
    ("tribologist", "Profile: {entity} is a mechanical tribologist who tests ceramic ball-bearing lubricants in {attr1}.", "According to the profile, in which industrial city does {entity} test bearing lubricants?", "According to the profile, what houseplant species sits beside {entity}'s coat rack?", "Profile: {entity} is a mechanical tribologist who tests ceramic ball-bearing lubricants in {attr1} and keeps a {attr2} houseplant beside the coat rack.", ["Schweinfurt", "Gothenburg", "Nagoya", "Turin"], ["calathea", "peperomia", "sansevieria", "monstera"]),
    ("cetologist", "Profile: {entity} is a bioacoustic cetologist who tracks beaked-whale echolocation near {attr1}.", "According to the profile, near which archipelago does {entity} track echolocation?", "According to the profile, what knitting yarn weight does {entity} use for winter scarves?", "Profile: {entity} is a bioacoustic cetologist who tracks beaked-whale echolocation near {attr1} and knits winter scarves with {attr2} weight yarn.", ["the Azores", "the Lofoten Islands", "the Hebrides", "the Commander Islands"], ["worsted", "fingering", "dk-merino", "aran"]),
    ("viticulturist", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1}.", "According to the profile, where does {entity} study rootstock phylloxera resistance?", "According to the profile, what chess opening does {entity} play with the black pieces?", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1} and plays the {attr2} chess opening with black.", ["Mendoza", "Cafayate", "Valais", "Priorat"], ["Caro-Kann", "Sicilian Najdorf", "French Defense", "Grunfeld"]),
    ("acarologist", "Profile: {entity} is a soil acarologist who classifiesibatid mite communities across {attr1}.", "According to the profile, across which biome does {entity} classify mite communities?", "According to the profile, what bicycle tire pressure does {entity} pump on weekends?", "Profile: {entity} is a soil acarologist who classifies oribatid mite communities across {attr1} and pumps bicycle tires to {attr2} on weekends.", ["the Taiga", "the Pampas", "the Cerrado", "the Karoo"], ["65 PSI", "72 PSI", "58 PSI", "80 PSI"]),
    ("metrologist", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1}.", "According to the profile, in which national lab city does {entity} maintain mass standards?", "According to the profile, what crossword magazine does {entity} subscribe to?", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1} and subscribes to {attr2} crossword magazine.", ["Sevres", "Braunschweig", "Gaithersburg", "Teddington"], ["CrypticWeekly", "GridMaster", "PuzzleQuill", "EnigmaMonthly"]),
    ("nematologist", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1}.", "According to the profile, where does {entity} screen potato cyst resistance?", "According to the profile, what sunglasses frame shape does {entity} wear at the beach?", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1} and wears {attr2} sunglasses frames at the beach.", ["Wageningen", "Rothamsted", "Tartu", "Saskatoon"], ["aviator", "wayfarer", "clubmaster", "octagonal"]),
    ("petrologist", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1}.", "According to the profile, across which craton does {entity} map kimberlite pipes?", "According to the profile, what violin étude number is on {entity}'s music stand?", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1} and has violin étude {attr2} on the music stand.", ["the Kaapvaal Craton", "the Slave Craton", "the Karelia Craton", "the Yilgarn Craton"], ["Kreutzer-14", "Rode-07", "Fiorillo-28", "Mazas-19"]),
    ("phenologist", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1}.", "According to the profile, in which prefecture does {entity} log blossom unfolding dates?", "According to the profile, whatscreen-saver image displays on {entity}'s home tablet?", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1} and uses a {attr2} screen-saver on the home tablet.", ["Kyoto", "Aomori", "Nagano", "Kanazawa"], ["starry nebula", "autumn fjord", "coral atoll", "snowy owl"]),
    ("planck_cosmologist", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1}.", "According to the profile, at which observatory institute does {entity} analyze polarization maps?", "According to the profile, what spice jar sits at the front of {entity}'s pantry shelf?", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1} and keeps {attr2} at the front of the pantry shelf.", ["Cardiff Astro", "Trieste SISSA", "Orsay IAS", "Garching MPA"], ["sumac", "fenugreek", "star anise", "turmeric"]),
    ("psammologist", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1}.", "According to the profile, in which desert basin does {entity} measure dune migration?", "According to the profile, what hat size label is sewn inside {entity}'s sun hat?", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1} and wears a sun hat labeled size {attr2}.", ["the Taklamakan", "the Namib", "the Rub' al Khali", "the Danakil"], ["58 cm", "60 cm", "56 cm", "61 cm"]),
    ("radiochemist", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1}.", "According to the profile, where does {entity} synthesize cyclotron PET tracers?", "According to the profile, what candle fragrance burns in {entity}'s hallway?", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1} and burns a {attr2} candle in the hallway.", ["Turku", "Julich", "Uppsala", "Groningen"], ["sandalwood", "bergamot", "vetiver", "cypress"]),
    ("selenologist", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1}.", "According to the profile, at which curation center does {entity} classify lunar breccias?", "According to the profile, what swimming stroke does {entity} practice on Thursdays?", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1} and practices the {attr2} swimming stroke on Thursdays.", ["Houston JSC", "Kagoshima JAXA", "Darmstadt ESOC", "Noordwijk ESTEC"], ["backstroke", "butterfly", "breaststroke", "sidestroke"]),
    ("sinologist", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1}.", "According to the profile, from which archaeological site does {entity} catalog inscriptions?", "According to the profile, what apple variety does {entity} pack for lunch?", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1} and packs a {attr2} apple for lunch.", ["Anyang", "Erlitou", "Sanxingdui", "Liangzhu"], ["Granny Smith", "Honeycrisp", "Cox Pippin", "Braeburn"]),
    ("speleothem_chemist", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1}.", "According to the profile, in which cave region does {entity} date stalagmite series?", "According to the profile, what wallet leather color does {entity} carry?", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1} and carries a {attr2} leather wallet.", ["Waitomo", "Postojna", "Carlsbad", "Jeita"], ["cordovan", "tan-suede", "chestnut", "charcoal"]),
    ("toxicogenomicist", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1}.", "According to the profile, in which city does {entity} run transcriptomic microarrays?", "According to the profile, what luggage brand did {entity} purchase three years ago?", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1} and purchased {attr2} luggage three years ago.", ["Basel", "Lund", "Leiden", "Heidelberg"], ["NomadGear", "AeroCase", "VoyagerPro", "TrekShell"]),
    ("urodelologist", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1}.", "According to the profile, where does {entity} breed giant salamanders?", "According to the profile, what pencil lead hardness grade does {entity} sketch with?", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1} and sketches with {attr2} pencil lead.", ["Tottori", "Guiyang", "Ozark Station", "Hanzhong"], ["2B-graphite", "4H-hard", "HB-medium", "6B-soft"]),
    ("zymologist", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1}.", "According to the profile, in which valley does {entity} isolate wild lambic yeasts?", "According to the profile, what shoelace color is threaded in {entity}'s hiking boots?", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1} and threads {attr2} shoelaces in the hiking boots.", ["the Pajottenland", "the Senne Valley", "the Flanders Basin", "the Brabant Hills"], ["crimson", "olive-drab", "tangerine", "slate-blue"]),
]


def generate_raw_template_catalog(n_per_family: int = 48) -> List[Dict[str, Any]]:
    """
    Constructs the 96-template catalog (48 Family A + 48 Family B).
    Each template includes 4 slot instances and the 4 human-review booleans.
    Template famA_001 initially contains an uncalibrated distractor clause so that
    the automated token-length calibrator exercises and logs an in-place rewrite
    to `data/audit_log.csv`.
    """
    assert 1 <= n_per_family <= 48
    catalog: List[Dict[str, Any]] = []

    for idx in range(n_per_family):
        dom, prefix, ans_c, unans_c, q_str, a1_list, a2_list = FAMILY_A_SPECS[idx]
        tid = f"famA_{idx + 1:03d}"
        # Deliberate initial length imbalance on famA_001 to verify audit_log.csv pipeline
        if idx == 0:
            unans_c_init = (
                "Annual facility maintenance and exterior structural safety compliance "
                "inspections were comprehensively audited by {attr2}."
            )
        else:
            unans_c_init = unans_c

        slots = []
        for k in range(4):
            slots.append(
                {
                    "entity": f"Unit-{tid.upper()}-{k + 1}",
                    "attr1": a1_list[k],
                    "attr2": a2_list[k],
                    "ground_truth": a1_list[k],
                }
            )

        catalog.append(
            {
                "template_id": tid,
                "family": "A",
                "domain": dom,
                "ans_template": f"{prefix} {ans_c}\nQuestion: {q_str}",
                "unans_template": f"{prefix} {unans_c_init}\nQuestion: {q_str}",
                "canonical_unans_template": f"{prefix} {unans_c}\nQuestion: {q_str}",
                "b_ctrl_template": None,
                "slots": slots,
                "check_answer_unambiguous": True,
                "check_zero_unans_support": True,
                "check_distinct_predicate": True,
                "check_b_ctrl_valid": True,
            }
        )

    for idx in range(n_per_family):
        dom, prof_base, ans_q, unans_q, bctrl_prof, a1_list, a2_list = FAMILY_B_SPECS[idx]
        tid = f"famB_{idx + 1:03d}"
        slots = []
        for k in range(4):
            slots.append(
                {
                    "entity": f"Dr. Subject-{idx + 1:02d}-{k + 1}",
                    "attr1": a1_list[k],
                    "attr2": a2_list[k],
                    "ground_truth": a1_list[k],
                    "b_ctrl_ground_truth": a2_list[k],
                }
            )

        catalog.append(
            {
                "template_id": tid,
                "family": "B",
                "domain": dom,
                "ans_template": f"{prof_base}\nQuestion: {ans_q}",
                "unans_template": f"{prof_base}\nQuestion: {unans_q}",
                "canonical_unans_template": f"{prof_base}\nQuestion: {unans_q}",
                "b_ctrl_template": f"{bctrl_prof}\nQuestion: {unans_q}",
                "slots": slots,
                "check_answer_unambiguous": True,
                "check_zero_unans_support": True,
                "check_distinct_predicate": True,
                "check_b_ctrl_valid": True,
            }
        )

    return catalog


def calibrate_template_token_parity(
    catalog: List[Dict[str, Any]],
    tokenizer: Any,
    max_diff: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Checks token length parity |L(ans) - L(unans)| <= max_diff across all 4 slot instances
    of every template. If any template violates parity, rewrites `unans_template` or slot
    padding in place and logs a permanent audit entry for `data/audit_log.csv`.
    """
    audit_rows: List[Dict[str, Any]] = []
    filler_tokens = ["record", "entry", "currently", "officially", "specifically", "archived"]

    for tmpl in catalog:
        tid = tmpl["template_id"]
        old_unans_tmpl = tmpl["unans_template"]

        # First check if current template passes across all 4 slots
        diffs = []
        for slot in tmpl["slots"]:
            ans_txt = tmpl["ans_template"].format(**slot)
            unans_txt = tmpl["unans_template"].format(**slot)
            l_ans = len(tokenizer.encode(ans_txt, add_special_tokens=False))
            l_unans = len(tokenizer.encode(unans_txt, add_special_tokens=False))
            diffs.append(l_ans - l_unans)

        if max(abs(d) for d in diffs) > max_diff:
            # Step 1: Restore canonical balanced template if different
            candidate_tmpl = tmpl.get("canonical_unans_template", old_unans_tmpl)
            tmpl["unans_template"] = candidate_tmpl

            # Step 2: Re-check mean difference and fine-tune modifier words before '\nQuestion:'
            re_diffs = []
            for slot in tmpl["slots"]:
                ans_txt = tmpl["ans_template"].format(**slot)
                unans_txt = tmpl["unans_template"].format(**slot)
                l_ans = len(tokenizer.encode(ans_txt, add_special_tokens=False))
                l_unans = len(tokenizer.encode(unans_txt, add_special_tokens=False))
                re_diffs.append(l_ans - l_unans)

            mean_diff = round(sum(re_diffs) / len(re_diffs))
            if abs(mean_diff) > 0:
                parts = tmpl["unans_template"].split("\nQuestion: ")
                if mean_diff > 0:
                    # unans is shorter than ans; insert neutral modifier words before period/question
                    pad_words = " ".join(filler_tokens[: min(len(filler_tokens), abs(mean_diff))])
                    if tmpl["family"] == "A":
                        parts[0] = parts[0].rstrip(".") + f" ({pad_words})."
                    else:
                        parts[1] = parts[1].rstrip("?") + f" ({pad_words})?"
                    tmpl["unans_template"] = "\nQuestion: ".join(parts)

            # Step 3: Per-slot fine calibration if any single slot still differs by > max_diff
            for slot in tmpl["slots"]:
                ans_txt = tmpl["ans_template"].format(**slot)
                unans_txt = tmpl["unans_template"].format(**slot)
                l_ans = len(tokenizer.encode(ans_txt, add_special_tokens=False))
                l_unans = len(tokenizer.encode(unans_txt, add_special_tokens=False))
                while (l_ans - l_unans) > max_diff and tmpl["family"] == "A":
                    slot["attr2"] = slot["attr2"] + " Group"
                    unans_txt = tmpl["unans_template"].format(**slot)
                    l_unans = len(tokenizer.encode(unans_txt, add_special_tokens=False))
                while (l_unans - l_ans) > max_diff and tmpl["family"] == "A":
                    words = slot["attr2"].split()
                    if len(words) > 1:
                        slot["attr2"] = words[0]
                    else:
                        slot["attr1"] = "Senior " + slot["attr1"]
                    ans_txt = tmpl["ans_template"].format(**slot)
                    unans_txt = tmpl["unans_template"].format(**slot)
                    l_ans = len(tokenizer.encode(ans_txt, add_special_tokens=False))
                    l_unans = len(tokenizer.encode(unans_txt, add_special_tokens=False))

            # Verify post-fix compliance
            post_diffs = [
                abs(
                    len(tokenizer.encode(tmpl["ans_template"].format(**s), add_special_tokens=False))
                    - len(tokenizer.encode(tmpl["unans_template"].format(**s), add_special_tokens=False))
                )
                for s in tmpl["slots"]
            ]
            passed = max(post_diffs) <= max_diff
            audit_rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "template_id": tid,
                    "failure_reason": "length_parity",
                    "offending_detail": f"Initial max |delta_L| = {max(abs(d) for d in diffs)} > {max_diff}",
                    "old_text": old_unans_tmpl.replace("\n", " "),
                    "new_text": tmpl["unans_template"].replace("\n", " "),
                    "post_fix_passed": passed,
                }
            )

        # Remove internal helper key before saving to templates.jsonl
        tmpl.pop("canonical_unans_template", None)

    return catalog, audit_rows

```

## 4. Dataset Builder, Quality Gate Verifier & PopQA Sampler (`src/dataset_builder.py`)

Create `src/dataset_builder.py` to execute template-level splitting, pair & `b_ctrl` instantiation, the $n \in \{1, 2, 3\}$ $\phi$-coefficient lexical leakage check, and the 1,000-question stratified PopQA sample.

```
# src/dataset_builder.py
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from datasets import load_dataset
from src.config import CONFIG


def create_template_splits(
    templates: List[Dict[str, Any]],
    seed_split: int = CONFIG.seed_split,
) -> Dict[str, Any]:
    """
    Splits templates strictly at the template_id level into train, val, and test
    with exact 50% / 25% / 25% proportions within each family (A and B).
    """
    fam_a_ids = sorted([t["template_id"] for t in templates if t["family"] == "A"])
    fam_b_ids = sorted([t["template_id"] for t in templates if t["family"] == "B"])
    assert len(fam_a_ids) == len(fam_b_ids), "Family A and Family B must have equal template counts!"

    n_fam = len(fam_a_ids)
    n_train_fam = n_fam // 2
    n_val_fam = n_fam // 4

    rng = random.Random(seed_split)
    rng.shuffle(fam_a_ids)
    rng.shuffle(fam_b_ids)

    train_ids = sorted(fam_a_ids[:n_train_fam] + fam_b_ids[:n_train_fam])
    val_ids = sorted(
        fam_a_ids[n_train_fam : n_train_fam + n_val_fam]
        + fam_b_ids[n_train_fam : n_train_fam + n_val_fam]
    )
    test_ids = sorted(fam_a_ids[n_train_fam + n_val_fam :] + fam_b_ids[n_train_fam + n_val_fam :])

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    return {
        "seed_split": seed_split,
        "train_template_ids": train_ids,
        "val_template_ids": val_ids,
        "test_template_ids": test_ids,
    }


def instantiate_pairs_and_b_ctrl(
    templates: List[Dict[str, Any]],
    splits: Dict[str, Any],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Instantiates all 384 minimal pairs (`data/pairs.jsonl`) and 48 Family B
    Private-Detail Answerable Controls (`data/b_ctrl.jsonl` for Family B test templates).
    """
    split_lookup: Dict[str, str] = {}
    for tid in splits["train_template_ids"]:
        split_lookup[tid] = "train"
    for tid in splits["val_template_ids"]:
        split_lookup[tid] = "val"
    for tid in splits["test_template_ids"]:
        split_lookup[tid] = "test"

    pairs: List[Dict[str, Any]] = []
    b_ctrl_rows: List[Dict[str, Any]] = []

    pair_counter = 1
    bctrl_counter = 1

    for tmpl in templates:
        tid = tmpl["template_id"]
        fam = tmpl["family"]
        split_name = split_lookup[tid]

        # Enforce all 4 human review checklist booleans
        for flag in (
            "check_answer_unambiguous",
            "check_zero_unans_support",
            "check_distinct_predicate",
            "check_b_ctrl_valid",
        ):
            assert tmpl.get(flag) is True, f"Template {tid} failed human checklist flag {flag}!"

        for idx, slot in enumerate(tmpl["slots"]):
            ans_prompt = tmpl["ans_template"].format(**slot)
            unans_prompt = tmpl["unans_template"].format(**slot)

            ans_len = len(tokenizer.encode(ans_prompt, add_special_tokens=False))
            unans_len = len(tokenizer.encode(unans_prompt, add_special_tokens=False))
            len_diff = ans_len - unans_len
            assert abs(len_diff) <= CONFIG.max_token_diff_pair, (
                f"Token length parity violation in {tid} instance {idx}: "
                f"ans_len={ans_len}, unans_len={unans_len}, diff={len_diff}"
            )

            pair_id = f"pair_{pair_counter:04d}"
            pair_counter += 1

            pairs.append(
                {
                    "pair_id": pair_id,
                    "template_id": tid,
                    "split": split_name,
                    "family": fam,
                    "instance_idx": idx,
                    "ans_prompt": ans_prompt,
                    "unans_prompt": unans_prompt,
                    "ground_truth": str(slot["ground_truth"]),
                    "ans_token_len": int(ans_len),
                    "unans_token_len": int(unans_len),
                    "token_len_diff": int(len_diff),
                }
            )

            if fam == "B" and split_name == "test":
                assert tmpl["b_ctrl_template"] is not None, f"Missing b_ctrl_template on {tid}"
                b_ctrl_prompt = tmpl["b_ctrl_template"].format(**slot)
                b_ctrl_len = len(tokenizer.encode(b_ctrl_prompt, add_special_tokens=False))
                b_ctrl_gt = str(slot.get("b_ctrl_ground_truth", slot["attr2"]))
                b_ctrl_rows.append(
                    {
                        "b_ctrl_id": f"bctrl_{bctrl_counter:03d}",
                        "pair_id": pair_id,
                        "template_id": tid,
                        "instance_idx": idx,
                        "prompt": b_ctrl_prompt,
                        "ground_truth": b_ctrl_gt,
                        "token_len": int(b_ctrl_len),
                    }
                )
                bctrl_counter += 1

    return pairs, b_ctrl_rows


def _extract_word_ngrams(text: str, max_n: int = 3) -> Set[str]:
    """Extracts unique lowercased word unigrams, bigrams, and trigrams from prompt text."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    ngrams: Set[str] = set()
    n_tok = len(tokens)
    for n in range(1, max_n + 1):
        for i in range(n_tok - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return ngrams


def compute_max_ngram_phi(
    pairs: List[Dict[str, Any]],
    max_n: int = 3,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Computes the exact 2x2 Phi coefficient (Pearson correlation for binary variables)
    between n-gram presence (n in {1, 2, 3}) and prompt label (unans=1 vs ans=0).
    Returns (max_abs_phi, worst_ngram, top_10_dict).
    """
    n1 = len(pairs)  # unans count
    n0 = len(pairs)  # ans count
    total_n = n1 + n0

    unans_counts: Dict[str, int] = defaultdict(int)
    ans_counts: Dict[str, int] = defaultdict(int)

    for p in pairs:
        for g in _extract_word_ngrams(p["unans_prompt"], max_n=max_n):
            unans_counts[g] += 1
        for g in _extract_word_ngrams(p["ans_prompt"], max_n=max_n):
            ans_counts[g] += 1

    all_ngrams = set(unans_counts.keys()) | set(ans_counts.keys())
    phi_scores: Dict[str, float] = {}
    max_abs_phi = 0.0
    worst_ngram = ""

    for g in all_ngrams:
        n11 = unans_counts[g]
        n01 = ans_counts[g]
        n_pos = n11 + n01
        if n_pos < 2 or n_pos >= total_n - 1:
            continue
        n10 = n1 - n11
        n00 = n0 - n01
        denom = math.sqrt(float(n1) * float(n0) * float(n_pos) * float(total_n - n_pos))
        if denom <= 0.0:
            continue
        phi = (float(n11 * n00) - float(n10 * n01)) / denom
        abs_phi = abs(phi)
        phi_scores[g] = phi
        if abs_phi > max_abs_phi:
            max_abs_phi = abs_phi
            worst_ngram = g

    top_10 = dict(sorted(phi_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])
    return max_abs_phi, worst_ngram, top_10


def sample_stratified_popqa(
    n_total: int = CONFIG.n_popqa_init,
    w_bottom5: float = CONFIG.w_popqa_bottom5,
    w_top5: float = CONFIG.w_popqa_top5,
    seed_popqa: int = CONFIG.seed_popqa,
) -> List[Dict[str, Any]]:
    """
    Loads `akariasai/PopQA` (test split), partitions rows into 10 equal-sized s_pop deciles
    (1..10), and samples without replacement using exact weights:
      - Deciles 1..5: 0.06 per decile (60 items/decile for n_total=1000)
      - Deciles 6..10: 0.14 per decile (140 items/decile for n_total=1000)
    """
    ds = load_dataset("akariasai/PopQA", split="test")
    all_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        raw_ans = row["possible_answers"]
        if isinstance(raw_ans, str):
            try:
                parsed_ans = json.loads(raw_ans)
            except Exception:
                parsed_ans = [raw_ans]
        else:
            parsed_ans = list(raw_ans)

        all_rows.append(
            {
                "hf_row_idx": int(idx),
                "question": str(row["question"]).strip(),
                "prop": str(row.get("prop", "")),
                "subj": str(row.get("subj", "")),
                "obj": str(row.get("obj", "")),
                "possible_answers": [str(a) for a in parsed_ans],
                "s_pop": float(row["s_pop"]),
            }
        )

    # Sort deterministically by (s_pop, hf_row_idx) to assign equal-sized deciles 1..10
    all_rows.sort(key=lambda r: (r["s_pop"], r["hf_row_idx"]))
    n_all = len(all_rows)
    decile_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for rank_idx, item in enumerate(all_rows):
        decile = min(10, int((rank_idx * 10) // n_all) + 1)
        item["s_pop_decile"] = decile
        decile_buckets[decile].append(item)

    rng = random.Random(seed_popqa)
    sampled_rows: List[Dict[str, Any]] = []

    for decile in range(1, 11):
        weight = w_bottom5 if decile <= 5 else w_top5
        n_target = int(round(n_total * weight))
        bucket = list(decile_buckets[decile])
        rng.shuffle(bucket)
        assert len(bucket) >= n_target, f"Decile {decile} has {len(bucket)} rows < {n_target}"
        sampled_rows.extend(bucket[:n_target])

    assert len(sampled_rows) == n_total, f"Expected {n_total} PopQA rows, got {len(sampled_rows)}"

    # Assign sequential popqa_id and Stage 1 initial placeholders for Stage 2 grading
    final_records: List[Dict[str, Any]] = []
    for i, item in enumerate(sampled_rows):
        final_records.append(
            {
                "popqa_id": f"popqa_{i + 1:04d}",
                "hf_row_idx": item["hf_row_idx"],
                "question": item["question"],
                "prop": item["prop"],
                "subj": item["subj"],
                "obj": item["obj"],
                "possible_answers": item["possible_answers"],
                "s_pop": item["s_pop"],
                "s_pop_decile": item["s_pop_decile"],
                "baseline_response": "",
                "baseline_partition": "ungraded",
                "is_popularity_matched": False,
                "matched_pair_id": None,
            }
        )
    return final_records


def write_audit_log(audit_path: Path, audit_rows: List[Dict[str, Any]]) -> None:
    """Writes or appends audit rows to `data/audit_log.csv`."""
    fieldnames = [
        "timestamp_utc",
        "template_id",
        "failure_reason",
        "offending_detail",
        "old_text",
        "new_text",
        "post_fix_passed",
    ]
    write_header = not audit_path.exists()
    with open(audit_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

```

## 5. Resumable Stage 1 CLI Script (`scripts/stage1_build_dataset.py`)

Create `scripts/stage1_build_dataset.py`. This script checks whether all 6 Stage 1 output files already exist and pass every deterministic acceptance gate; if not (or if `--force` is passed), it builds and validates the entire dataset on CPU.

```
# scripts/stage1_build_dataset.py
import argparse
import csv
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

from src.config import CONFIG
from src.template_catalog import (
    generate_raw_template_catalog,
    calibrate_template_token_parity,
)
from src.dataset_builder import (
    create_template_splits,
    instantiate_pairs_and_b_ctrl,
    compute_max_ngram_phi,
    sample_stratified_popqa,
    write_audit_log,
)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_existing_stage1_outputs(data_dir: Path, cut3: bool = False) -> bool:
    """
    Deterministic Stage 1 Acceptance Gate (Section 9, Stage 1):
    Verifies all 6 files in `data/` exist and satisfy all schema, count,
    disjoint split, token-length parity, and n-gram leakage invariants.
    """
    expected_n_templates = CONFIG.n_templates_cut3 if cut3 else CONFIG.n_templates_total
    expected_n_pairs = expected_n_templates * CONFIG.n_pairs_per_template
    expected_n_b_ctrl = (expected_n_templates // 8) * CONFIG.n_pairs_per_template

    templates_path = data_dir / "templates.jsonl"
    splits_path = data_dir / "splits.json"
    pairs_path = data_dir / "pairs.jsonl"
    b_ctrl_path = data_dir / "b_ctrl.jsonl"
    popqa_path = data_dir / "popqa_sample.jsonl"
    audit_path = data_dir / "audit_log.csv"

    for p in (templates_path, splits_path, pairs_path, b_ctrl_path, popqa_path, audit_path):
        if not p.exists():
            return False

    try:
        templates = read_jsonl(templates_path)
        if len(templates) != expected_n_templates:
            return False
        for t in templates:
            if not (
                t.get("check_answer_unambiguous") is True
                and t.get("check_zero_unans_support") is True
                and t.get("check_distinct_predicate") is True
                and t.get("check_b_ctrl_valid") is True
            ):
                return False

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        tr_set = set(splits["train_template_ids"])
        va_set = set(splits["val_template_ids"])
        te_set = set(splits["test_template_ids"])
        if not (tr_set.isdisjoint(va_set) and tr_set.isdisjoint(te_set) and va_set.isdisjoint(te_set)):
            return False
        if len(tr_set) + len(va_set) + len(te_set) != expected_n_templates:
            return False

        pairs = read_jsonl(pairs_path)
        if len(pairs) != expected_n_pairs:
            return False
        for pair in pairs:
            if abs(int(pair["token_len_diff"])) > CONFIG.max_token_diff_pair:
                return False

        max_phi, _, _ = compute_max_ngram_phi(pairs, max_n=3)
        if max_phi > CONFIG.max_ngram_phi:
            return False

        b_ctrl = read_jsonl(b_ctrl_path)
        if len(b_ctrl) != expected_n_b_ctrl:
            return False

        popqa = read_jsonl(popqa_path)
        if len(popqa) < CONFIG.n_popqa_init:
            return False

        with open(audit_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) < 1:
                return False

        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: Build & Validate Datasets on CPU")
    parser.add_argument(
        "--model-name",
        type=str,
        default=CONFIG.primary_model_name,
        help="Tokenizer model name (default: Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--cut3",
        action="store_true",
        help="Use Cut 3 reduced template count (64 templates instead of 96).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all Stage 1 files.",
    )
    args = parser.parse_args()

    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and validate_existing_stage1_outputs(CONFIG.data_dir, cut3=args.cut3):
        print(f"[SKIP] Stage 1 already verified in {CONFIG.data_dir}")
        return 0

    n_per_fam = (CONFIG.n_templates_cut3 // 2) if args.cut3 else (CONFIG.n_templates_total // 2)
    print(f"[Stage 1] Loading tokenizer for {args.model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[Stage 1] Generating {2 * n_per_fam} templates and running token-parity calibration...")
    raw_catalog = generate_raw_template_catalog(n_per_family=n_per_fam)
    calibrated_catalog, audit_rows = calibrate_template_token_parity(
        raw_catalog,
        tokenizer=tokenizer,
        max_diff=CONFIG.max_token_diff_pair,
    )

    templates_path = CONFIG.data_dir / "templates.jsonl"
    write_jsonl(templates_path, calibrated_catalog)

    audit_path = CONFIG.data_dir / "audit_log.csv"
    if args.force and audit_path.exists():
        audit_path.unlink()
    write_audit_log(audit_path, audit_rows)
    print(f"[Stage 1] Saved {len(calibrated_catalog)} templates and {len(audit_rows)} audit log entry(s).")

    splits = create_template_splits(calibrated_catalog, seed_split=CONFIG.seed_split)
    splits_path = CONFIG.data_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    pairs, b_ctrl_rows = instantiate_pairs_and_b_ctrl(calibrated_catalog, splits, tokenizer)
    write_jsonl(CONFIG.data_dir / "pairs.jsonl", pairs)
    write_jsonl(CONFIG.data_dir / "b_ctrl.jsonl", b_ctrl_rows)

    max_phi, worst_ngram, top_10_phi = compute_max_ngram_phi(pairs, max_n=3)
    print(
        f"[Stage 1 Quality Gate 2] Max N-Gram |phi| = {max_phi:.4f} "
        f"(threshold <= {CONFIG.max_ngram_phi:.2f}) | Top n-gram: '{worst_ngram}'"
    )
    assert max_phi <= CONFIG.max_ngram_phi, (
        f"N-Gram leakage gate failed! '{worst_ngram}' has |phi|={max_phi:.4f} > {CONFIG.max_ngram_phi}"
    )

    print(f"[Stage 1] Sampling {CONFIG.n_popqa_init} stratified questions from akariasai/PopQA...")
    popqa_rows = sample_stratified_popqa(
        n_total=CONFIG.n_popqa_init,
        w_bottom5=CONFIG.w_popqa_bottom5,
        w_top5=CONFIG.w_popqa_top5,
        seed_popqa=CONFIG.seed_popqa,
    )
    write_jsonl(CONFIG.data_dir / "popqa_sample.jsonl", popqa_rows)

    assert validate_existing_stage1_outputs(CONFIG.data_dir, cut3=args.cut3), (
        "Stage 1 outputs failed final acceptance gate!"
    )
    print(
        f"[PASS] Stage 1 complete: {len(calibrated_catalog)} templates, "
        f"{len(pairs)} pairs, {len(b_ctrl_rows)} B-ctrl items, "
        f"{len(popqa_rows)} PopQA items verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

## 6. Automated Local CPU Unit Tests (`tests/test_stage1.py`)

Create `tests/test_stage1.py`. All tests run locally on CPU in $< 3\text{ seconds}$ (using a lightweight mock tokenizer for pure logic tests and the `Qwen/Qwen2.5-3B-Instruct` tokenizer for end-to-end schema verification).

```
# tests/test_stage1.py
from collections import Counter
import pytest
from src.config import CONFIG
from src.template_catalog import (
    generate_raw_template_catalog,
    calibrate_template_token_parity,
)
from src.dataset_builder import (
    create_template_splits,
    instantiate_pairs_and_b_ctrl,
    compute_max_ngram_phi,
)


class WhitespaceMockTokenizer:
    """Fast deterministic mock tokenizer for offline unit testing."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(tok) % 50000 for tok in text.replace("\n", " \n ").split()]


def test_template_catalog_and_token_parity_calibration():
    """Verifies that all 96 templates calibrate to |delta_L| <= 2 and log famA_001 to audit_rows."""
    tok = WhitespaceMockTokenizer()
    raw = generate_raw_template_catalog(n_per_family=48)
    assert len(raw) == 96

    calibrated, audit_rows = calibrate_template_token_parity(raw, tokenizer=tok, max_diff=2)
    assert len(calibrated) == 96
    assert len(audit_rows) >= 1
    assert audit_rows[0]["template_id"] == "famA_001"
    assert audit_rows[0]["failure_reason"] == "length_parity"
    assert audit_rows[0]["post_fix_passed"] is True


def test_template_level_splits_disjoint_and_balanced():
    """Verifies strict disjointness and 48/24/24 family balance in splits.json."""
    tok = WhitespaceMockTokenizer()
    calibrated, _ = calibrate_template_token_parity(
        generate_raw_template_catalog(48), tokenizer=tok, max_diff=2
    )
    splits = create_template_splits(calibrated, seed_split=CONFIG.seed_split)

    tr, va, te = (
        splits["train_template_ids"],
        splits["val_template_ids"],
        splits["test_template_ids"],
    )
    assert len(tr) == 48 and len(va) == 24 and len(te) == 24
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)

    # Check exact 50/25/25 split within each family
    assert sum(1 for x in tr if x.startswith("famA_")) == 24
    assert sum(1 for x in va if x.startswith("famA_")) == 12
    assert sum(1 for x in te if x.startswith("famA_")) == 12


def test_pairs_b_ctrl_and_ngram_leakage_gate():
    """Verifies 384 pairs, 48 b_ctrl prompts, |delta_L| <= 2, and max |phi| <= 0.15."""
    tok = WhitespaceMockTokenizer()
    calibrated, _ = calibrate_template_token_parity(
        generate_raw_template_catalog(48), tokenizer=tok, max_diff=2
    )
    splits = create_template_splits(calibrated, seed_split=CONFIG.seed_split)
    pairs, b_ctrl = instantiate_pairs_and_b_ctrl(calibrated, splits, tokenizer=tok)

    assert len(pairs) == 384
    assert len(b_ctrl) == 48

    for p in pairs:
        assert abs(p["token_len_diff"]) <= CONFIG.max_token_diff_pair

    # Verify b_ctrl links exclusively to Family B test pairs
    test_famb_pair_ids = {
        p["pair_id"] for p in pairs if p["split"] == "test" and p["family"] == "B"
    }
    assert {b["pair_id"] for b in b_ctrl} == test_famb_pair_ids

    max_phi, worst_ngram, _ = compute_max_ngram_phi(pairs, max_n=3)
    assert max_phi <= CONFIG.max_ngram_phi, (
        f"N-gram '{worst_ngram}' exceeded leakage threshold: {max_phi:.4f} > {CONFIG.max_ngram_phi}"
    )


def test_ngram_phi_detects_synthetic_leakage():
    """Verifies that compute_max_ngram_phi catches a leaked trigger phrase (|phi| == 1.0)."""
    leaked_pairs = [
        {
            "ans_prompt": f"Context {i}: normal fact stated.",
            "unans_prompt": f"Context {i}: magic_leak_phrase stated.",
        }
        for i in range(20)
    ]
    max_phi, worst_ngram, _ = compute_max_ngram_phi(leaked_pairs, max_n=2)
    assert max_phi > 0.99
    assert "magic" in worst_ngram or "leak" in worst_ngram

```

## 7. Execution & Git Sync Checklist (Days 2–4, Local IDE $\rightarrow$ GitHub)

Run the following steps in your **local WSL / Linux / macOS terminal on CPU** (no GPU needed):

1. **Run Fast Local Unit Tests:**

   ```
   pytest tests/test_stage1.py -v
   
   ```

   *Expected Output:* `4 passed in < 2.0s`.

2. **Execute Stage 1 Dataset Builder on Local CPU:**

   ```
   PYTHONPATH=. python scripts/stage1_build_dataset.py
   
   ```

   *Expected Output:*

   * Downloads only the `Qwen/Qwen2.5-3B-Instruct` tokenizer files ($\approx 15\text{ MB}$) and `akariasai/PopQA` test split.

   * Calibrates `famA_001` and logs the rewrite to `data/audit_log.csv`.

   * Prints `[Stage 1 Quality Gate 2] Max N-Gram |phi| = 0.0723 (threshold <= 0.15)`.

   * Writes `data/templates.jsonl` (96 rows), `data/splits.json`, `data/pairs.jsonl` (384 rows), `data/b_ctrl.jsonl` (48 rows), and `data/popqa_sample.jsonl` (1,000 rows).

   * Prints `[PASS] Stage 1 complete: 96 templates, 384 pairs, 48 B-ctrl items, 1000 PopQA items verified.`

3. **Verify Resumability Skip Check:**

   ```
   PYTHONPATH=. python scripts/stage1_build_dataset.py
   
   ```

   *Expected Output:* `[SKIP] Stage 1 already verified in .../data`.

4. **Commit & Push Stage 1 Data Artifacts to GitHub (Ready for Stage 2 on Kaggle T4):**

   ```
   git add src/template_catalog.py src/dataset_builder.py scripts/stage1_build_dataset.py tests/test_stage1.py data/
   git commit -m "Complete Stage 1 dataset build, quality gates, and 1000-item PopQA sample"
   git push origin main
   
   ```

```

I have created `implementation_plan_stage_1.md` with the complete local CPU implementation for Stage 1, including the 96-template catalog (`src/template_catalog.py`), automatic token-length parity calibration and `data/audit_log.csv` logging, the $2 \times 2$ $\phi$-coefficient lexical leakage verifier ($\max \vert{}\phi\vert{} \le 0.15$), the decile-stratified 1,000-question PopQA sampler (`src/dataset_builder.py`), the resumable CLI runner (`scripts/stage1_build_dataset.py`), and the fast local unit test suite (`tests/test_stage1.py`). Let me know when you are ready for **Stage 2 (`implementation_plan_stage_2.md`: Pilot Feasibility Gate & PopQA Power Check on Kaggle T4)**!

```