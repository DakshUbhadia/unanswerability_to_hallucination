# src/template_catalog.py
from datetime import datetime, timezone
import random
from functools import lru_cache
from typing import Any, Dict, List, Tuple
from src.config import CONFIG


_GIVEN_NAMES = ("Mara", "Elias", "Nadia", "Rohan", "Leila", "Tomas", "Amina", "Jonas", "Priya", "Caleb", "Sofia", "Kenji", "Amara", "Felix", "Ines", "Omar", "Elena", "Hugo", "Mei", "Darius", "Anika", "Simon", "Farah", "Luca")
_SURNAMES = ("Ellison", "Halvorsen", "Mercer", "Okafor", "Varga", "Chen", "Navarro", "Bennett", "Kowalski", "Rahman", "Fischer", "Sato", "Moreau", "Patel", "Lind", "Reyes", "Brooks", "Nakamura", "Mensah", "Weber", "Costa", "Shah", "Walsh", "Kim", "Anders", "Bauer", "Carter", "Duarte", "Evans", "Foster", "Grant", "Hughes", "Ivanov", "Jensen", "Keller", "Lopez", "Mori", "Nolan", "Olsen", "Parker", "Quinn", "Rossi", "Singh", "Turner", "Ueda", "Voss", "Wong", "Young")
_ENTITY_WORDS = ("Halvorsen", "Meridian", "Aster", "Northstar", "Cypress", "Solstice", "Juniper", "Westhaven", "Larkspur", "Borealis", "Silverleaf", "Crestwood", "Redwood", "Evergreen", "Stonebridge", "Bluehaven", "Cedar", "Windward", "Kingswell", "Riverside", "Fairview", "Brightwater", "Ashford", "Willow")
_TECHNICAL_DISTRACTORS = (
    "The instrument calibration", "The diagnostic algorithm", "The reference dataset", "The validation protocol",
    "The signal processing transform", "The materials assay", "The system reliability report", "The statistical design",
    "The sensor alignment procedure", "The environmental measurement", "The equipment performance benchmark", "The safety envelope",
    "The molecular assay", "The imaging reconstruction", "The computational simulation", "The structural response",
    "The field sampling protocol", "The archival classification", "The network performance benchmark", "The control system test",
    "The power conversion analysis", "The thermal reaction profile", "The acoustic measurement", "The geospatial surface",
    "The chemical composition assay", "The biological reaction analysis", "The optical transmission test", "The fluid dynamics solution",
    "The manufacturing tolerance assessment", "The reliability forecast", "The data quality assessment", "The numerical solver",
    "The calibration dataset", "The technical risk analysis", "The process optimization curve", "The laboratory control protocol",
    "The spectral decomposition", "The coral band isotope ratio reconstruction", "The measurement uncertainty estimate", "The systems integration test",
    "The material behavior profile", "The trajectory optimization", "The detailed diagnostic imaging protocol", "The experimental design",
    "The quality assurance protocol", "The engineering feasibility study", "The signal classification model", "The instrument validation study",
)
_TECHNICAL_VERBS = (
    "calibrated", "modeled", "validated", "benchmarked", "simulated", "quantified", "characterized", "profiled",
    "optimized", "estimated", "measured", "mapped", "classified", "tested", "diagnosed", "parameterized",
    "analyzed", "assessed", "documented", "monitored", "reconstructed", "integrated", "compared", "audited",
    "verified", "derived", "resolved", "computed", "sampled", "corrected", "interpreted", "forecast",
    "fitted", "examined", "operated", "designed", "reviewed", "recorded", "refined", "tracked",
    "surveyed", "interpolated", "extrapolated", "correlated", "sequenced", "aggregated", "segmented", "projected",
)


def _fixed_names() -> List[str]:
    names = [f"{first} {last}" for first in _GIVEN_NAMES for last in _SURNAMES]
    random.Random(CONFIG.seed_global).shuffle(names)
    return names


def _human_names_a() -> List[str]:
    return _fixed_names()[:192]


def _human_names_b() -> List[str]:
    return _fixed_names()[192:384]


def _token_matched_human_names(tokenizer: Any) -> Tuple[List[str], List[str]]:
    """Keep Family A answer names and pair each distractor name at equal tokenizer length."""
    names = _fixed_names()
    answer_names = names[:192]
    # Keep the middle block reserved for Family B person entities.
    distractor_pool = names[192:384] + names[576:]
    lengths = {name: len(tokenizer.encode(name, add_special_tokens=False)) for name in distractor_pool}
    used = set()
    distractor_names: List[str] = []
    for i, answer_name in enumerate(answer_names):
        target_len = len(tokenizer.encode(answer_name, add_special_tokens=False))
        existing = names[192 + i]
        if lengths[existing] == target_len and existing not in used:
            chosen = existing
        else:
            chosen = next((n for n in distractor_pool if n not in used and lengths[n] == target_len), None)
            if chosen is None:
                raise ValueError(f"Name pools cannot provide a distinct {target_len}-token distractor name")
        used.add(chosen)
        distractor_names.append(chosen)
    return answer_names, distractor_names


@lru_cache(maxsize=1)
def _natural_entities() -> Tuple[str, ...]:
    words = [f"the {a} {b} Initiative" for a in _ENTITY_WORDS for b in _ENTITY_WORDS]
    random.Random(CONFIG.seed_global).shuffle(words)
    selected = words[:192]
    used = set()
    for pos, name in enumerate(selected):
        parts = name.removeprefix("the ").removesuffix(" Initiative").split()
        if len(parts) == 2 and parts[0] == parts[1] or name in used:
            replacement = next(candidate for candidate in words[192:]
                               if candidate not in used and candidate not in selected
                               and candidate.removeprefix("the ").removesuffix(" Initiative").split()[0]
                               != candidate.removeprefix("the ").removesuffix(" Initiative").split()[1])
            selected[pos] = replacement
            name = replacement
        used.add(name)
    return tuple(selected)


def _natural_entity(idx: int, k: int) -> str:
    return _natural_entities()[idx * 4 + k]


def _natural_person(idx: int, k: int) -> str:
    return _fixed_names()[384 + idx * 4 + k]


# 48 distinct domains, answerable clauses, and distractor clauses for Family A.
# Each tuple: (domain, context_prefix, ans_clause, unans_clause, question_str, attr1_examples, attr2_examples)
# Both clauses share the "... was <verb> by {attr}." skeleton and the question is identical across
# ans / unans, so only template-unique content words differ between the two prompts (Quality Gate 2).
# attr1 names carry no honorifics: a title such as "Dr." would appear only in answerable prompts.
FAMILY_A_SPECS: List[Tuple[str, str, str, str, str, List[str], List[str]]] = [
    ("solar_research", "Clinical and engineering dossier for {entity} archived in 2019.", "Primary technical oversight was led by {attr1}.", "Annual facility maintenance was audited by {attr2}.", "Based strictly on the dossier, who led primary technical oversight for {entity}?", ["Aris Thorne", "Nadia Lin", "Kenji Sato", "Elena Rostova"], ["Apex Compliance", "Vertex Audits", "Beacon Review", "Crest Inspectors"]),
    ("geothermal_grid", "Infrastructure dispatch record for {entity} logged in 2020.", "The chief turbine architect on record was {attr1}.", "The regional concrete supplier on record was {attr2}.", "Based strictly on the dispatch record, who was the chief turbine architect for {entity}?", ["Marcus Vale", "Lukas Weber", "Ravi Kumar", "Soren Lindqvist"], ["Nordic Cement", "Boreal Stone", "Glacier Aggregate", "Fjord Materials"]),
    ("quantum_ledger", "Cryptography validation memo for {entity} filed in 2021.", "The lattice key protocol was authored by {attr1}.", "The cooling rack enclosure was shipped by {attr2}.", "Based strictly on the validation memo, who authored the lattice key protocol for {entity}?", ["Kofi Mensah", "Paul Grant", "Sven Olsen", "Devon Brooks"], ["CryoLogistics", "Polar Freight", "Thermal Transit", "Alps Cargo"]),
    ("solid_battery", "Electrochemistry lab bulletin for {entity} issued in 2022.", "The electrolyte dopant formula was synthesized by {attr1}.", "The upstairs warehouse ventilation duct was cleaned by {attr2}.", "Based strictly on the lab bulletin, who synthesized the electrolyte dopant formula for {entity}?", ["Yuna Park", "Mateo Silva", "Priya Nair", "Felix Bauer"], ["CleanAir Partners", "DuctPro Team", "Breeze Maintenance", "AeroScrub"]),
    ("genomic_biobank", "Bioinformatics registry entry for {entity} updated in 2018.", "The sequencing pipeline calibration was verified by {attr1}.", "The exterior parking lot paving was completed by {attr2}.", "Based strictly on the registry entry, who verified the sequencing pipeline calibration for {entity}?", ["Hannah Cole", "Omar Farooq", "Celine Moreau", "Pavel Novak"], ["Metro Asphalt", "PaveRight", "Tarmac Brothers", "Roadway Partners"]),
    ("photonic_chip", "Semiconductor lithography note for {entity} recorded in 2023.", "The waveguide etching mask was designed by {attr1}.", "The staff cafeteria vending contract was managed by {attr2}.", "Based strictly on the lithography note, who designed the waveguide etching mask for {entity}?", ["Ingrid Holm", "Rohan Mehta", "Dante Moretti", "Keiko Tanaka"], ["SnackLine", "Refresh Vendors", "BistroMatic", "Pantry Express"]),
    ("maritime_manifest", "Port authority customs log for {entity} stamped in 2017.", "The vessel navigation charter was signed by {attr1}.", "The crane repainting was billed by {attr2}.", "Based strictly on the customs log, who signed the vessel navigation charter for {entity}?", ["Arthur Pendleton", "Sofia Reyes", "Henrik Olsen", "Mei-Ling Zhou"], ["Harbor Coatings", "Marine Paint", "Pier Tints", "Seawall Coat"]),
    ("clinical_trial", "Pharmacology study protocol for {entity} registered in 2020.", "The biostatistics endpoint model was reviewed by {attr1}.", "The courier specimen cold-chain packaging was supplied by {attr2}.", "Based strictly on the study protocol, who reviewed the biostatistics endpoint model for {entity}?", ["Julian Mercer", "Ananya Rao", "Tobias Klein", "Rachel Stern"], ["SafeBox Medical", "ParcelShield", "ColdPack Containers", "BioCrate"]),
    ("satellite_telemetry", "Orbital tracking summary for {entity} transmitted in 2021.", "The attitude thruster burn sequence was calculated by {attr1}.", "The ground station lawn mowing was scheduled by {attr2}.", "Based strictly on the tracking summary, who calculated the attitude thruster burn sequence for {entity}?", ["Liam O'Connor", "Mei Chen", "Sergei Volkov", "Fatima Zahra"], ["GreenTurf", "Valley Landscaping", "Meadow Grounds", "Horizon Mowers"]),
    ("municipal_zoning", "City planning commission minutes for {entity} archived in 2019.", "The seismic variance assessment was certified by {attr1}.", "The microphone rental was arranged by {attr2}.", "Based strictly on the commission minutes, who certified the seismic variance assessment for {entity}?", ["Victor Morales", "Diane Foster", "Haruki Abe", "Nora Lind"], ["StageSound Rentals", "CityAudio Gear", "Echo AV Systems", "ClearTone"]),
    ("museum_provenance", "Curatorial acquisition ledger for {entity} cataloged in 2016.", "The radiocarbon pigment analysis was conducted by {attr1}.", "The marble lobby lighting fixture was polished by {attr2}.", "Based strictly on the acquisition ledger, who conducted the radiocarbon pigment analysis for {entity}?", ["Beatrice Vane", "Lorenzo Conti", "Nina Park", "Hugo Martin"], ["Lumen Polishers", "Gleam Maintenance", "Gallery Brass", "BrightFixture"]),
    ("supply_audit", "Procurement compliance report for {entity} finalized in 2022.", "The cobalt smelter traceability check was performed by {attr1}.", "The quarterly office stationery printing run was fulfilled by {attr2}.", "Based strictly on the compliance report, who performed the cobalt smelter traceability check for {entity}?", ["Laura Klein", "Gavin Thorne", "Satoshi Endo", "Mark Hill"], ["PaperTrail Press", "InkWell Stationers", "PageOne Print", "FineLeaf Paper"]),
    ("archaeology_dig", "Stratigraphy field journal for {entity} compiled in 2018.", "The obsidian blade classification was documented by {attr1}.", "The basecamp tent waterproofing was treated by {attr2}.", "Based strictly on the field journal, who documented the obsidian blade classification for {entity}?", ["Nadia Lin", "Jan Kowalski", "Nikos Pappas", "Aisha Bello"], ["CanvasGuard", "Outback Shelters", "TrekProof Tents", "DryCamp Gear"]),
    ("incident_postmortem", "Cloud reliability postmortem for {entity} published in 2023.", "The BGP routing table rollback was executed by {attr1}.", "The breakroom espresso machine was descaled by {attr2}.", "Based strictly on the postmortem, who executed the BGP routing table rollback for {entity}?", ["Raj Patel", "Olga Ivanova", "Marco Bellini", "Grace Liu"], ["BeanSteam", "CafeCare Techs", "RoastRepair", "BrewFix"]),
    ("patent_abstract", "Intellectual property docket for {entity} indexed in 2020.", "The prior art patentability search was completed by {attr1}.", "The archival binder embossing was invoiced by {attr2}.", "Based strictly on the docket, who completed the prior art patentability search for {entity}?", ["Omar Haddad", "Lea Fontaine", "Rajeev Kapoor", "Astrid Lindholm"], ["GoldLeaf Bindery", "Crest Embossing", "Folio Crafts", "SpinePress"]),
    ("weather_bulletin", "Atmospheric sonde bulletin for {entity} logged in 2021.", "The tropopause wind shear profile was calibrated by {attr1}.", "The access road snow plowing was contracted by {attr2}.", "Based strictly on the sonde bulletin, who calibrated the tropopause wind shear profile for {entity}?", ["Tariq Aziz", "Katrin Weiss", "Daichi Mori", "Yuki Mori"], ["FrostBlade Plows", "Alpine Clearing", "WinterRoad", "IcePath"]),
    ("aviation_inspection", "Airframe maintenance logbook for {entity} signed in 2019.", "The ultrasonic wing spar test was inspected by {attr1}.", "The hangar floor epoxy coating was applied by {attr2}.", "Based strictly on the maintenance logbook, who inspected the ultrasonic wing spar test for {entity}?", ["Gregory Walsh", "Alina Popescu", "Kenzo Fujii", "Carlos Mendez"], ["Dura Floors", "HangarCoat", "PolySeal Surfaces", "AeroDeck Finish"]),
    ("hydrology_survey", "Watershed sediment survey for {entity} recorded in 2017.", "The dissolved nitrate titration was analyzed by {attr1}.", "The field trailer tire replacement was serviced by {attr2}.", "Based strictly on the sediment survey, who analyzed the dissolved nitrate titration for {entity}?", ["Amara Obi", "Cheng Wei", "Lars Mikkelsen", "Felix Wong"], ["AxlePro Tires", "TrailWheel Garage", "Ridgeway Auto", "Summit Tread"]),
    ("fusion_tokamak", "Plasma confinement log for {entity} archived in 2022.", "The divertor magnetic coil alignment was tuned by {attr1}.", "The visitor badge lanyard order was shipped by {attr2}.", "Based strictly on the confinement log, who tuned the divertor magnetic coil alignment for {entity}?", ["Ilan Gur", "Veronika Horvath", "Taro Kurosawa", "Quentin Blake"], ["BadgeWorks", "LanyardPlus", "IDTag Express", "CordCraft"]),
    ("botanical_herbarium", "Taxonomic voucher sheet for {entity} dated in 2018.", "The pollen morphology microscopy was verified by {attr1}.", "The cabinet silica gel was restocked by {attr2}.", "Based strictly on the voucher sheet, who verified the pollen morphology microscopy for {entity}?", ["Ingrid Dahl", "Farah Malik", "Aiko Hayashi", "Bernard Mercier"], ["DryPack Gel", "SilicaSafe", "MoistureGuard", "AridBox"]),
    ("seismic_array", "Tectonic borehole record for {entity} filed in 2020.", "The downhole geophone gain was adjusted by {attr1}.", "The perimeter chainlink fence was repaired by {attr2}.", "Based strictly on the borehole record, who adjusted the downhole geophone gain for {entity}?", ["Malcolm Reed", "Leo Russo", "Hana Kim", "Erik Lund"], ["IronLink Fencing", "BorderWire", "SteelPost Builders", "SecureMesh"]),
    ("vaccine_coldchain", "Biologics stability manifest for {entity} audited in 2021.", "The ultra-low freezer thermocouple was validated by {attr1}.", "The loading dock rubber bumper was bolted by {attr2}.", "Based strictly on the stability manifest, who validated the ultra-low freezer thermocouple for {entity}?", ["Tessa Rowe", "Nikhil Sharma", "Emilie Blanc", "Petra Novak"], ["DockGuard Rubber", "BayBumper", "ImpactBuffers", "QuayHardware"]),
    ("subsea_cable", "Transoceanic fiber log for {entity} compiled in 2019.", "The optical repeater attenuation test was run by {attr1}.", "The galley kitchen dishwasher was repaired by {attr2}.", "Based strictly on the fiber log, who ran the optical repeater attenuation test for {entity}?", ["Declan Byrne", "Zara Khan", "Mateus Rocha", "Freja Holm"], ["GalleyFix", "ShipAppliance", "DeckWasher Tech", "NautiRepair"]),
    ("radiotelescope", "Interferometry baseline note for {entity} logged in 2023.", "The maser atomic clock phase was locked by {attr1}.", "The gravel walkway weed trimming was done by {attr2}.", "Based strictly on the baseline note, who locked the maser atomic clock phase for {entity}?", ["Gareth Hughes", "Ximena Soto", "Hiro Tanaka", "Leila Abbasi"], ["DesertTrim", "PathClear Grounds", "Mesa Brush", "AridScape"]),
    ("glacier_core", "Cryosphere drilling report for {entity} registered in 2018.", "The trapped methane isotope ratio was measured by {attr1}.", "The generator exhaust muffler was welded by {attr2}.", "Based strictly on the drilling report, who measured the trapped methane isotope ratio for {entity}?", ["Astrid Nyman", "Bruno Silva", "Haruka Sano", "Samira Khan"], ["ArcWeld Polar", "TorchCraft Metal", "MufflerFix Shop", "IronSeam Works"]),
    ("rail_signaling", "Interlocking safety certificate for {entity} issued in 2020.", "The wayside transponder fail-safe logic was tested by {attr1}.", "The platform station bench wood staining was finished by {attr2}.", "Based strictly on the safety certificate, who tested the wayside transponder fail-safe logic for {entity}?", ["Roland Beck", "Carmen Vega", "Amy Walsh", "Lydia Park"], ["OakGlow Finishes", "TimberStain", "BenchCraft Wood", "CedarCoat"]),
    ("wind_offshore", "Nacelle telemetry dossier for {entity} recorded in 2022.", "The pitch bearing vibration spectrum was diagnosed by {attr1}.", "The lifejacket locker was inventoried by {attr2}.", "Based strictly on the telemetry dossier, who diagnosed the pitch bearing vibration spectrum for {entity}?", ["Willem de Boer", "Chiara Ferri", "Jun-Ho Seo", "Maeve Kelly"], ["SeaSafe Gear", "BuoyLocker Audits", "DeckVest", "OceanSafety"]),
    ("desalination_plant", "Reverse osmosis logbook for {entity} archived in 2021.", "The membrane permeate conductivity was checked by {attr1}.", "The guardhouse window tinting was installed by {attr2}.", "Based strictly on the osmosis logbook, who checked the membrane permeate conductivity for {entity}?", ["Tariq Nassar", "Arjun Mehta", "Kazuki Miura", "Sonja Berg"], ["SunBlock Glass", "SolarShade Film", "ClearTint Pros", "GlareGuard"]),
    ("acoustics_chamber", "Anechoic calibration sheet for {entity} filed in 2019.", "The fiberglass wedge absorption curve was plotted by {attr1}.", "The hallway coat rack mounting was anchored by {attr2}.", "Based strictly on the calibration sheet, who plotted the fiberglass wedge absorption curve for {entity}?", ["Elif Demir", "Diego Ruiz", "Shohei Kanda", "Sara Cole"], ["WallAnchor Handyman", "StudMount", "HallwayFix", "BoltRight"]),
    ("mycology_culture", "Fungal strain repository card for {entity} dated in 2020.", "The agar mycelium growth rate was scored by {attr1}.", "The autoclave bay floor drain was flushed by {attr2}.", "Based strictly on the repository card, who scored the agar mycelium growth rate for {entity}?", ["Callum Ross", "Beatriz Santos", "Natsuki Aoki", "Bilal Rahman"], ["PipeClear Plumbing", "DrainJet", "FlowRight Techs", "CleanTrap"]),
    ("nuclear_dosimetry", "Health physics badge report for {entity} certified in 2017.", "The neutron track etch calibration was approved by {attr1}.", "The reception desk floral display was watered by {attr2}.", "Based strictly on the badge report, who approved the neutron track etch calibration for {entity}?", ["Ines Duarte", "Paulina Sikora", "Ren Takahashi", "Marta Ruiz"], ["PetalCare Flora", "LobbyBloom", "GreenStem", "VaseFresh"]),
    ("autonomous_lidar", "Sensor fusion test log for {entity} recorded in 2023.", "The point-cloud extrinsic matrix was solved by {attr1}.", "The garage roll-up door chain was greased by {attr2}.", "Based strictly on the test log, who solved the point-cloud extrinsic matrix for {entity}?", ["Kenji Ito", "Sofia Lindqvist", "Matteo Galli", "Hina Matsui"], ["RollSmooth Doors", "ChainLube Garage", "OverheadFix", "BayDoor"]),
    ("metallurgy_foundry", "Alloy tensile certificate for {entity} stamped in 2018.", "The grain boundary microhardness was measured by {attr1}.", "The locker room soap dispenser was refilled by {attr2}.", "Based strictly on the tensile certificate, who measured the grain boundary microhardness for {entity}?", ["Gunter Braun", "Joana Pinto", "Tom Reid", "Ivan Petrov"], ["SudsSupply", "WashRoom Hygiene", "CleanHands", "SoapLine Direct"]),
    ("volcanology_gas", "Fumarole plume bulletin for {entity} logged in 2021.", "The sulfur dioxide flux spectrometer was operated by {attr1}.", "The field jeep windshield wiper was replaced by {attr2}.", "Based strictly on the plume bulletin, who operated the sulfur dioxide flux spectrometer for {entity}?", ["Marco Rossi", "Keiko Mori", "Samuel Osei", "Freya Lind"], ["ClearView Auto", "WiperPro Shop", "GlassCare Garage", "TrailAuto Parts"]),
    ("cryptanalysis_archive", "Ciphertext recovery memo for {entity} cataloged in 2016.", "The frequency digram substitution table was reconstructed by {attr1}.", "The reading room velvet curtain was dry-cleaned by {attr2}.", "Based strictly on the recovery memo, who reconstructed the frequency digram substitution table for {entity}?", ["Mila Horvat", "Marta Kowalczyk", "Shinji Ikeda", "Clara Morel"], ["VelvetCare Cleaners", "DraperyWash", "FineFabric Press", "CurtainFresh"]),
    ("aquaculture_hatchery", "Marine larval tank record for {entity} filed in 2022.", "The dissolved oxygen biofilter loop was balanced by {attr1}.", "The boat ramp algae pressure-washing was done by {attr2}.", "Based strictly on the tank record, who balanced the dissolved oxygen biofilter loop for {entity}?", ["Haakon Lund", "Camila Rojas", "Yuto Nakajima", "Amina Bello"], ["JetWash Marine", "RampScrub Pros", "HydroBlast", "Slipway Cleaners"]),
    ("avionics_firmware", "Flight control verification note for {entity} issued in 2020.", "The watchdog timer interrupt latency was profiled by {attr1}.", "The conference table chair casters were oiled by {attr2}.", "Based strictly on the verification note, who profiled the watchdog timer interrupt latency for {entity}?", ["Anders Holm", "Celine Garnier", "Kaito Sasaki", "Ines Silva"], ["GlideCaster", "OfficeChair Fix", "SwivelSmooth", "SeatRoll"]),
    ("paleoclimate_coral", "Carbonate sclerochronology sheet for {entity} archived in 2019.", "The strontium-to-calcium paleothermometer was run by {attr1}.", "The second-floor lab sink faucet aerator screen was descaled by {attr2}.", "Based strictly on the sclerochronology sheet, who ran the strontium-to-calcium paleothermometer for {entity}?", ["Georgia Webb", "Rafael Mendes", "Li Wei", "Anton Lind"], ["TapFlow Plumbing", "AeratorClean", "SinkFix", "ClearSpout"]),
    ("superconductor_wire", "Cryogenic magnet test sheet for {entity} logged in 2023.", "The critical current quench threshold was recorded by {attr1}.", "The shipping pallet shrink-wrap film was ordered by {attr2}.", "Based strictly on the test sheet, who recorded the critical current quench threshold for {entity}?", ["Lev Landau-Smith", "Giulia Costa", "Haruto Kato", "Sara Lindstrom"], ["WrapTight Packaging", "PalletFilm", "ShrinkGuard", "CrateWrap Direct"]),
    ("epigraphy_tablet", "Cuneiform conservation log for {entity} compiled in 2017.", "The kiln-fired clay desalination bath was monitored by {attr1}.", "The glass display case brass padlock was lubricated by {attr2}.", "Based strictly on the conservation log, who monitored the kiln-fired clay desalination bath for {entity}?", ["Cyrus Vane", "Ana Silva", "Kenji Abe", "Miriam Levy"], ["LockGlide Security", "BrassKey", "PadlockPro", "SafeHinge"]),
    ("forestry_lidar", "Canopy biomass inventory for {entity} finalized in 2021.", "The allometric crown diameter regression was fitted by {attr1}.", "The cabin woodstove chimney sweep was performed by {attr2}.", "Based strictly on the biomass inventory, who fitted the allometric crown diameter regression for {entity}?", ["Jan Novak", "Elsa Lindqvist", "Tomas Varga", "Mei-Hui Lin"], ["SootFree", "FlueClean Pros", "HearthCare", "ChimneyBrush"]),
    ("cleanroom_hvac", "Particulate certification log for {entity} signed in 2020.", "The HEPA laminar airflow velocity was certified by {attr1}.", "The upstairs hallway recycling bin signage was printed by {attr2}.", "Based strictly on the certification log, who certified the HEPA laminar airflow velocity for {entity}?", ["Bernard Quist", "Ruth Adler", "Theo Grant", "Paula Castro"], ["BinTag Graphics", "EcoSign Print", "LabelCraft Shop", "SortSticker"]),
    ("sonar_bathymetry", "Multibeam trench survey for {entity} recorded in 2018.", "The sound velocity thermocline profile was corrected by {attr1}.", "The deck winch canvas cover was stitched by {attr2}.", "Based strictly on the trench survey, who corrected the sound velocity thermocline profile for {entity}?", ["Alistair Boyd", "Marina Silva", "Koji Yamamoto", "Ingrid Holm"], ["SailStitch Canvas", "DeckCover Tailor", "TarpSeam Marine", "WinchHood"]),
    ("enzymology_assay", "Kinetics spectrophotometry log for {entity} dated in 2022.", "The Michaelis constant substrate curve was fitted by {attr1}.", "The cold-room door magnetic gasket was wiped by {attr2}.", "Based strictly on the spectrophotometry log, who fitted the Michaelis constant substrate curve for {entity}?", ["Vera Novak", "Luis Ortega", "Rina Hashimoto", "Samuel Osei"], ["SealWipe", "GasketClean", "DoorStrip", "CoolSeal"]),
    ("bridge_strain", "Structural viaduct inspection for {entity} filed in 2019.", "The suspension cable acoustic emission was logged by {attr1}.", "The tollbooth traffic cones were stacked by {attr2}.", "Based strictly on the viaduct inspection, who logged the suspension cable acoustic emission for {entity}?", ["Eva Novak", "Elena Popa", "Daichi Goto", "Tomas Novak"], ["ConeStack Traffic", "LaneMarker", "RoadSafety Supply", "PylonCare"]),
    ("astrometry_transit", "Exoplanet photometry circular for {entity} published in 2023.", "The limb-darkening transit lightcurve was modeled by {attr1}.", "The dome shutter motor chain was tensioned by {attr2}.", "Based strictly on the photometry circular, who modeled the limb-darkening transit lightcurve for {entity}?", ["Miranda Fell", "Lucas Meyer", "Takuya Fujita", "Lena Ford"], ["DomeDrive Mechanics", "ShutterChain", "GearTension Tech", "ObservatoryFix"]),
    ("toxicology_elisa", "Forensic assay worksheet for {entity} archived in 2020.", "The microplate optical density curve was verified by {attr1}.", "The biohazard bin foot pedal was tightened by {attr2}.", "Based strictly on the assay worksheet, who verified the microplate optical density curve for {entity}?", ["Dario Conti", "Ana Lopez", "Yuji Murakami", "Mira Shah"], ["PedalFix Hardware", "LabBin", "StepLid Repair", "CanHinge"]),
    ("piezoelectric_sensor", "Ferroelectric thin-film note for {entity} recorded in 2021.", "The poling hysteresis loop voltage was measured by {attr1}.", "The outbound shipping crate foam padding insert was cut to size by {attr2}.", "Based strictly on the thin-film note, who measured the poling hysteresis loop voltage for {entity}?", ["Cedric Moss", "Adriana Rossi", "Noah Price", "Linnea Berg"], ["FoamCut Packaging", "CushionBox", "InsertCraft", "DieCut Foam"]),
]

# Replace legacy facility/vendor distractors with distinct technical work by a person.
_FAMILY_A_MANUAL_CLAUSES = {
    0: 'The solar degradation study was led by {attr2}.',
    1: 'The geothermal well monitoring array was designed by {attr2}.',
    2: 'The entropy-source test specification was authored by {attr2}.',
    3: 'The silicon-anode binder polymer was synthesized by {attr2}.',
    4: 'The variant expression matrix was verified by {attr2}.',
    5: 'The silicon wafer etch-depth model was designed by {attr2}.',
    6: 'The cargo vessel stability calculation was signed by {attr2}.',
    7: 'The planned dose escalation schedule was reviewed by {attr2}.',
    8: 'The satellite orbit decay rate was calculated by {attr2}.',
    9: 'The floodplain setback map was certified by {attr2}.',
    10: 'The ceramic glaze abrasion test was conducted by {attr2}.',
    11: 'The cobalt cathode corrosion test was performed by {attr2}.',
    12: 'The pottery shard typology was documented by {attr2}.',
    13: 'The distributed network service failover procedure was executed by {attr2}.',
    14: 'The trademark similarity search was completed by {attr2}.',
    15: 'The boundary-layer temperature sensor array was calibrated by {attr2}.',
    16: 'The avionics wiring harness was inspected by {attr2}.',
    17: 'The watershed surface runoff model was analyzed by {attr2}.',
    18: 'The tokamak plasma confinement feedback gain was tuned by {attr2}.',
    19: 'The seed viability count was verified by {attr2}.',
    20: 'The fault slip-rate model parameters were adjusted by {attr2}.',
    21: 'The vaccine antigen potency curve was validated by {attr2}.',
    22: 'The subsea splice loss test was run by {attr2}.',
    23: 'The observatory telemetry encryption channel was locked by {attr2}.',
    24: 'The ice-core dust concentration was measured by {attr2}.',
    25: 'The rail wheel-bearing rolling-contact fatigue life was tested by {attr2}.',
    26: 'The yaw gearbox tooth wear was diagnosed by {attr2}.',
    27: 'The brine salt rejection rate was checked by {attr2}.',
    28: 'The room impulse response was plotted by {attr2}.',
    29: 'The fungal colony morphology from agar plates was scored by {attr2}.',
    30: 'The gamma-camera image reconstruction protocol was approved by {attr2}.',
    31: 'The autonomous vehicle emergency-braking trajectory was solved by {attr2}.',
    32: 'The alloy phase volume fraction was measured by {attr2}.',
    33: 'The volcanic ash particle classifier was operated by {attr2}.',
    34: 'The public-key certificate revocation sequence was reconstructed by {attr2}.',
    35: 'The aquaculture tank salinity was balanced by {attr2}.',
    36: 'The avionics bus latency was profiled by {attr2}.',
    37: 'The reef sediment carbonate dissolution assay was run by {attr2}.',
    38: 'The superconducting sample temperature ramp was recorded by {attr2}.',
    39: 'The fired-clay tablet mineral composition was monitored by {attr2}.',
    40: 'The canopy height regression model was fitted by {attr2}.',
    41: 'The cleanroom airborne particle count was certified by {attr2}.',
    42: 'The seabed backscatter mosaic was corrected by {attr2}.',
    43: 'The enzyme inhibition kinetic model was fitted by {attr2}.',
    44: 'The bridge deck strain profile was logged by {attr2}.',
    45: 'The stellar radial-velocity curve was modeled by {attr2}.',
    46: 'The metabolite dose-response assay was verified by {attr2}.',
    47: 'The piezoelectric resonance frequency response was measured by {attr2}.',
}
_FAMILY_A_ANSWER_OVERRIDES = {
    1: 'The geothermal turbine layout was designed by {attr1}.',
    37: 'The coral growth-band isotope analysis was run by {attr1}.',
}
_FAMILY_A_QUESTION_OVERRIDES = {
    1: 'Based strictly on the dispatch record, who designed the geothermal turbine layout for {entity}?',
    37: 'Based strictly on the sclerochronology sheet, who ran the coral growth-band isotope analysis for {entity}?',
}
FAMILY_A_SPECS = [
    (dom, prefix, _FAMILY_A_ANSWER_OVERRIDES.get(i, ans), _FAMILY_A_MANUAL_CLAUSES.get(i, f"{_TECHNICAL_DISTRACTORS[i]} was {_TECHNICAL_VERBS[i]} by {{attr2}}."), _FAMILY_A_QUESTION_OVERRIDES.get(i, question),
     _human_names_a()[i * 4 : i * 4 + 4], _human_names_b()[i * 4 : i * 4 + 4])
    for i, (dom, prefix, ans, _legacy_unans, question, a1, a2) in enumerate(FAMILY_A_SPECS)
]

# 48 distinct biographical/entity profile domains for Family B.
# Each tuple: (domain, profile, ans_question, unans_question, b_ctrl_profile, attr1_examples, attr2_examples)
# ans_question and unans_question share the same interrogative stem ("where does", "in which ... did",
# "what is the name of", ...) and differ only in the predicate, so no stem n-gram correlates with the label.
# b_ctrl_profile states the private detail ({attr2}) that unans_question asks about.
FAMILY_B_SPECS: List[Tuple[str, str, str, str, str, List[str], List[str]]] = [
    ("alpine_botanist", "Profile: {entity} is an alpine botanist whose doctoral fieldwork was completed in {attr1}.", "According to the profile, where was {entity}'s doctoral fieldwork completed?", "According to the profile, where was {entity}'s childhood hamster purchased?", "Profile: {entity} is an alpine botanist whose doctoral fieldwork was completed in {attr1} and whose childhood hamster was purchased in {attr2}.", ["Zurich", "Innsbruck", "Grenoble", "Bergen"], ["Lucerne", "Salzburg", "Annecy", "Stavanger"]),
    ("glaciologist", "Profile: {entity} is a polar glaciologist who joined the ice-core institute in {attr1}.", "According to the profile, in what year did {entity} join the ice-core institute?", "According to the profile, in what year did {entity} lose their first milk tooth?", "Profile: {entity} is a polar glaciologist who joined the ice-core institute in {attr1} and lost their first milk tooth in {attr2}.", ["1998", "2004", "2011", "2015"], ["1979", "1983", "1990", "1986"]),
    ("mycologist", "Profile: {entity} is a forest mycologist who lectures at the university in {attr1}.", "According to the profile, in which city does {entity} lecture?", "According to the profile, in which city does {entity} keep spare furniture?", "Profile: {entity} is a forest mycologist who lectures at the university in {attr1} and keeps spare furniture in {attr2}.", ["Portland", "Uppsala", "Sapporo", "Freiburg"], ["Salem", "Lund", "Otaru", "Basel"]),
    ("seismologist", "Profile: {entity} is a marine seismologist whose flagship research vessel is named {attr1}.", "According to the profile, what is the name of {entity}'s flagship research vessel?", "According to the profile, what is the name of {entity}'s childhood rocking horse?", "Profile: {entity} is a marine seismologist whose flagship research vessel is named {attr1} and whose childhood rocking horse was named {attr2}.", ["RV Calypso-II", "RV TrenchStar", "RV Nautilus-V", "RV Pelagic"], ["Thunder", "Biscuit", "Comet", "Saffron"]),
    ("numismatist", "Profile: {entity} is a classical numismatist specializing in bronze coinage from {attr1}.", "According to the profile, which region's bronze coinage does {entity} specialize in?", "According to the profile, which region's honey does {entity} stir into morning tea?", "Profile: {entity} is a classical numismatist specializing in bronze coinage from {attr1} who stirs honey from {attr2} into morning tea.", ["Thrace", "Bactria", "Lydia", "Seleucia"], ["Crete", "Tuscany", "Provence", "Anatolia"]),
    ("dendrochronologist", "Profile: {entity} is a dendrochronologist who curates the bristlecone archive in {attr1}.", "According to the profile, where does {entity} curate the bristlecone archive?", "According to the profile, where does {entity} park vintage scooters overnight?", "Profile: {entity} is a dendrochronologist who curates the bristlecone archive in {attr1} and parks vintage scooters overnight in {attr2}.", ["Flagstaff", "Reno", "Boise", "Bozeman"], ["Carson City", "Elko", "Pocatello", "Missoula"]),
    ("paleographer", "Profile: {entity} is a medieval paleographer who translates parchment charters written in {attr1}.", "According to the profile, which language are the charters {entity} translates written in?", "According to the profile, which language are the recipes {entity} inherited written in?", "Profile: {entity} is a medieval paleographer who translates parchment charters written in {attr1} and inherited family recipes written in {attr2}.", ["Old Occitan", "Church Slavonic", "Middle High German", "Hiberno-Latin"], ["Basque", "Yiddish", "Gaelic", "Romansh"]),
    ("bryologist", "Profile: {entity} is a tropical bryologist who cataloged liverwort specimens across {attr1}.", "According to the profile, across which island did {entity} catalog liverwort specimens?", "According to the profile, across which island did {entity} hike as a student?", "Profile: {entity} is a tropical bryologist who cataloged liverwort specimens across {attr1} and hiked across {attr2} as a student.", ["Borneo", "Madagascar", "Tasmania", "Socotra"], ["Crete", "Sicily", "Corsica", "Mallorca"]),
    ("astrophotographer", "Profile: {entity} is an astrophotographer who operates a robotic reflector in {attr1}.", "According to the profile, where does {entity} operate a robotic reflector?", "According to the profile, where does {entity} rent holiday cottages?", "Profile: {entity} is an astrophotographer who operates a robotic reflector in {attr1} and rents holiday cottages in {attr2}.", ["Atacama", "La Palma", "Mauna Kea", "Karoo"], ["Cornwall", "Brittany", "Algarve", "Kerry"]),
    ("ethnomusicologist", "Profile: {entity} is an ethnomusicologist who records polyphonic folk songs in {attr1}.", "According to the profile, where does {entity} record polyphonic folk songs?", "According to the profile, where does {entity} buy fresh sourdough bread?", "Profile: {entity} is an ethnomusicologist who records polyphonic folk songs in {attr1} and buys fresh sourdough bread in {attr2}.", ["Sardinia", "Epirus", "Svaneti", "Tuva"], ["Cagliari", "Ioannina", "Kutaisi", "Kyzyl"]),
    ("limnologist", "Profile: {entity} is a freshwater limnologist who monitors diatom blooms in {attr1}.", "According to the profile, in which lake does {entity} monitor diatom blooms?", "According to the profile, in which lake does {entity} swim on summer weekends?", "Profile: {entity} is a freshwater limnologist who monitors diatom blooms in {attr1} and swims in {attr2} on summer weekends.", ["Lake Baikal", "Lake Tanganyika", "Lake Ohrid", "Lake Biwa"], ["Lake Como", "Lake Bled", "Lake Annecy", "Lake Hallstatt"]),
    ("horologist", "Profile: {entity} is an antique horologist who restores marine chronometers built in {attr1}.", "According to the profile, from which century are the chronometers {entity} restores?", "According to the profile, from which century are the teacups {entity} inherited?", "Profile: {entity} is an antique horologist who restores marine chronometers built in {attr1} and inherited teacups made in {attr2}.", ["the 18th century", "the 17th century", "the 19th century", "the 16th century"], ["the 20th century", "the 19th century", "the 18th century", "the 17th century"]),
    ("lichenologist", "Profile: {entity} is an arctic lichenologist who directs the tundra station in {attr1}.", "According to the profile, where does {entity} direct the tundra station?", "According to the profile, where does {entity} store spare kayaks?", "Profile: {entity} is an arctic lichenologist who directs the tundra station in {attr1} and stores spare kayaks in {attr2}.", ["Svalbard", "Tromso", "Nuuk", "Abisko"], ["Bodo", "Kiruna", "Ilulissat", "Narvik"]),
    ("speleologist", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1}.", "According to the profile, in which province did {entity} map subterranean caverns?", "According to the profile, where did {entity} camp as a child?", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1} and attended summer camp in {attr2} as a child.", ["Guizhou", "Slovenia", "Yucatan", "Cantabria"], ["Hunan", "Istria", "Oaxaca", "Asturias"]),
    ("malacologist", "Profile: {entity} is a marine malacologist who studies cone snail venoms at {attr1}.", "According to the profile, at which aquarium does {entity} study cone snail venoms?", "According to the profile, at which aquarium does {entity} host family birthday parties?", "Profile: {entity} is a marine malacologist who studies cone snail venoms at {attr1} and hosts family birthday parties at {attr2}.", ["Monterey Lab", "Okinawa Marine", "Monaco Ocean", "Woods Hole"], ["Genoa Aquarium", "Osaka Kaiyukan", "Shedd Chicago", "Oceanario Lisbon"]),
    ("lepidopterist", "Profile: {entity} is a cloud-forest lepidopterist who tracks monarch migrations through {attr1}.", "According to the profile, through which mountain range does {entity} track migrations?", "According to the profile, through which mountain range does {entity} drive each spring?", "Profile: {entity} is a cloud-forest lepidopterist who tracks monarch migrations through {attr1} and drives through {attr2} each spring.", ["the Sierra Madre", "the Cordillera Real", "the Rwenzori", "the Nilgiri Hills"], ["the Dolomites", "the Pyrenees", "the Tatras", "the Atlas"]),
    ("gemologist", "Profile: {entity} is a forensic gemologist who grades sapphire inclusions in {attr1}.", "According to the profile, in which trading hub does {entity} grade sapphire inclusions?", "According to the profile, in which trading hub does {entity} buy birthday gifts?", "Profile: {entity} is a forensic gemologist who grades sapphire inclusions in {attr1} and buys birthday gifts in {attr2}.", ["Antwerp", "Geneva", "Jaipur", "Chanthaburi"], ["Dubai", "Bangkok", "Hong Kong", "Singapore"]),
    ("luthiery_maker", "Profile: {entity} is a concert luthier who carves cello soundboards from spruce grown in {attr1}.", "According to the profile, where is the spruce {entity} uses for cello soundboards grown?", "According to the profile, where is the lavender {entity} uses for scented bathwater grown?", "Profile: {entity} is a concert luthier who carves cello soundboards from spruce grown in {attr1} and uses lavender grown in {attr2} for scented bathwater.", ["Val di Fiemme", "Mittenwald", "Carpathia", "Sitka"], ["Provence", "Hokkaido", "Norfolk", "Tasmania"]),
    ("oenologist", "Profile: {entity} is a fermentation oenologist who consults for hillside vineyards in {attr1}.", "According to the profile, in which valley does {entity} consult for hillside vineyards?", "According to the profile, in which valley does {entity} spend quiet winter holidays?", "Profile: {entity} is a fermentation oenologist who consults for hillside vineyards in {attr1} and spends quiet winter holidays in {attr2}.", ["the Douro Valley", "the Mosel Valley", "the Barossa Valley", "the Colchagua Valley"], ["the Loire Valley", "the Napa Valley", "the Hunter Valley", "the Maipo Valley"]),
    ("carillonneur", "Profile: {entity} is a municipal carillonneur who performs weekly bell recitals in {attr1}.", "According to the profile, in which town does {entity} perform weekly bell recitals?", "According to the profile, in which town does {entity} grow allotment vegetables?", "Profile: {entity} is a municipal carillonneur who performs weekly bell recitals in {attr1} and grows allotment vegetables in {attr2}.", ["Mechelen", "Delft", "Bruges", "Utrecht"], ["Leuven", "Haarlem", "Ghent", "Gouda"]),
    ("archivometrics", "Profile: {entity} is a bibliometric historian who indexes scientific correspondence from {attr1}.", "According to the profile, from which decade does {entity} index scientific correspondence?", "According to the profile, from which decade does {entity} collect vinyl records?", "Profile: {entity} is a bibliometric historian who indexes scientific correspondence from {attr1} and collects vinyl records from {attr2}.", ["the 1890s", "the 1920s", "the 1870s", "the 1930s"], ["the 1960s", "the 1970s", "the 1980s", "the 1950s"]),
    ("palynologist", "Profile: {entity} is a quaternary palynologist who extracts fossilized spores from peat bogs in {attr1}.", "According to the profile, where does {entity} extract fossilized spores from peat bogs?", "According to the profile, where does {entity} rent sturdy hiking boots for long weekend trips?", "Profile: {entity} is a quaternary palynologist who extracts fossilized spores from peat bogs in {attr1} and rents sturdy hiking boots for long weekend trips in {attr2}.", ["Connemara", "Karelia", "Tierra del Fuego", "Hokkaido"], ["Galway", "Helsinki", "Ushuaia", "Sapporo"]),
    ("crystallographer", "Profile: {entity} is an X-ray crystallographer who operates beamline diffractometers at {attr1}.", "According to the profile, at which synchrotron facility does {entity} operate diffractometers?", "According to the profile, at which alpine ski resort does {entity} spend long winter holidays?", "Profile: {entity} is an X-ray crystallographer who operates beamline diffractometers at {attr1} and spends long winter holidays at the alpine ski resort {attr2}.", ["DESY Hamburg", "ESRF Grenoble", "SPring-8 Hyogo", "PSI Villigen"], ["Zermatt", "Chamonix", "Niseko", "Davos"]),
    ("ichthyologist", "Profile: {entity} is a deep-sea ichthyologist who describes bioluminescent lanternfish from {attr1}.", "According to the profile, from which ocean basin does {entity} describe lanternfish?", "According to the profile, from which bakery does {entity} order birthday cakes?", "Profile: {entity} is a deep-sea ichthyologist who describes bioluminescent lanternfish from {attr1} and orders birthday cakes from {attr2}.", ["the Banda Sea", "the Weddell Sea", "the Sargasso Sea", "the Coral Sea"], ["Rosemary Bakes", "Crumb Corner", "Golden Oven", "Sugarloaf Bakery"]),
    ("cartographer", "Profile: {entity} is a bathymetric cartographer who compiles nautical depth charts for {attr1}.", "According to the profile, for which strait does {entity} compile nautical depth charts?", "According to the profile, for which charity does {entity} knit warm woollen winter blankets?", "Profile: {entity} is a bathymetric cartographer who compiles nautical depth charts for {attr1} and knits warm woollen winter blankets for {attr2}.", ["the Luzon Strait", "the Denmark Strait", "the Bass Strait", "the Cook Strait"], ["Warm Hearts", "Blanket Bridge", "Kind Stitches", "Cozy Hands"]),
    ("herpetologist", "Profile: {entity} is a montane herpetologist who surveys glass-frog populations in {attr1}.", "According to the profile, in which reserve does {entity} survey glass-frog populations?", "According to the profile, in which park does {entity} walk the family beagle?", "Profile: {entity} is a montane herpetologist who surveys glass-frog populations in {attr1} and walks the family beagle in {attr2}.", ["Monteverde", "Choco Darien", "Manu Biosphere", "Yasuni Park"], ["Retiro Park", "Hyde Park", "Ueno Park", "Stanley Park"]),
    ("metallographer", "Profile: {entity} is an archaeometallurgist who analyzes Damascus steel blades housed in {attr1}.", "According to the profile, in which museum city are the blades {entity} analyzes housed?", "According to the profile, in which city are the paintings {entity} owns stored?", "Profile: {entity} is an archaeometallurgist who analyzes Damascus steel blades housed in {attr1} and owns paintings stored in {attr2}.", ["Istanbul", "Toledo", "Graz", "London"], ["Porto", "Lyon", "Krakow", "Bilbao"]),
    ("helminthologist", "Profile: {entity} is a veterinary parasitologist who teaches diagnostic microscopy in {attr1}.", "According to the profile, where does {entity} teach diagnostic microscopy?", "According to the profile, where does {entity} take evening salsa classes?", "Profile: {entity} is a veterinary parasitologist who teaches diagnostic microscopy in {attr1} and takes evening salsa classes in {attr2}.", ["Ghent", "Utrecht", "Davis", "Guelph"], ["Antwerp", "Rotterdam", "Sacramento", "Toronto"]),
    ("pomologist", "Profile: {entity} is an heirloom pomologist who grafts frost-hardy apple cultivars in {attr1}.", "According to the profile, in which region does {entity} graft frost-hardy apple cultivars?", "According to the profile, in which region does {entity} park holiday caravans?", "Profile: {entity} is an heirloom pomologist who grafts frost-hardy apple cultivars in {attr1} and parks holiday caravans in {attr2}.", ["Almaty", "Normandy", "the Finger Lakes", "South Tyrol"], ["Brittany", "Bavaria", "Vermont", "Lapland"]),
    ("rheologist", "Profile: {entity} is a polymer rheologist who measures viscoelastic shear flows at {attr1}.", "According to the profile, at which polytechnic does {entity} measure viscoelastic shear flows?", "According to the profile, at which neighborhood gym does {entity} train on early weekday mornings?", "Profile: {entity} is a polymer rheologist who measures viscoelastic shear flows at {attr1} and trains at the neighborhood gym {attr2} on early weekday mornings.", ["EPFL Lausanne", "TU Delft", "RWTH Aachen", "KTH Stockholm"], ["IronCore Gym", "PulseFit Club", "Summit Gym", "Atlas Fitness"]),
    ("sedimentologist", "Profile: {entity} is a deltaic sedimentologist who cores turbidite fans offshore of {attr1}.", "According to the profile, offshore of which delta does {entity} core turbidite fans?", "According to the profile, offshore of which island does {entity} go sailing with old friends?", "Profile: {entity} is a deltaic sedimentologist who cores turbidite fans offshore of {attr1} and goes sailing with old friends offshore of {attr2}.", ["the Bengal Delta", "the Rhone Delta", "the Mackenzie Delta", "the Danube Delta"], ["Gotland", "Mykonos", "Rhodes", "Bornholm"]),
    ("tribologist", "Profile: {entity} is a mechanical tribologist who tests ceramic ball-bearing lubricants in {attr1}.", "According to the profile, in which industrial city does {entity} test bearing lubricants?", "According to the profile, in which city does {entity} rent garden plots?", "Profile: {entity} is a mechanical tribologist who tests ceramic ball-bearing lubricants in {attr1} and rents garden plots in {attr2}.", ["Schweinfurt", "Gothenburg", "Nagoya", "Turin"], ["Bremen", "Aarhus", "Kobe", "Genoa"]),
    ("cetologist", "Profile: {entity} is a bioacoustic cetologist who tracks beaked-whale echolocation near {attr1}.", "According to the profile, near which archipelago does {entity} track echolocation?", "According to the profile, near which archipelago does {entity} spend family summers?", "Profile: {entity} is a bioacoustic cetologist who tracks beaked-whale echolocation near {attr1} and spends family summers near {attr2}.", ["the Azores", "the Lofoten Islands", "the Hebrides", "the Commander Islands"], ["the Cyclades", "the Aland Islands", "the Scilly Isles", "the Faroe Islands"]),
    ("viticulturist", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1}.", "According to the profile, where does {entity} study rootstock phylloxera resistance?", "According to the profile, where does {entity} play chess?", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1} and plays competitive weekend chess tournaments with friends in {attr2}.", ["Mendoza", "Cafayate", "Valais", "Priorat"], ["Rosario", "Salta", "Sion", "Tarragona"]),
    ("acarologist", "Profile: {entity} is a soil acarologist who classifies oribatid mite communities across {attr1}.", "According to the profile, across which biome does {entity} classify mite communities?", "According to the profile, across which country does {entity} cycle each summer?", "Profile: {entity} is a soil acarologist who classifies oribatid mite communities across {attr1} and cycles across {attr2} each summer.", ["the Taiga", "the Pampas", "the Cerrado", "the Karoo"], ["Portugal", "Ireland", "Austria", "Denmark"]),
    ("metrologist", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1}.", "According to the profile, in which national lab city does {entity} maintain mass standards?", "According to the profile, in which city does {entity} attend crossword club meetings?", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1} and attends crossword club meetings in {attr2}.", ["Sevres", "Braunschweig", "Gaithersburg", "Teddington"], ["Nantes", "Kassel", "Baltimore", "Oxford"]),
    ("nematologist", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1}.", "According to the profile, where does {entity} screen potato cyst resistance?", "According to the profile, which city does {entity} take weekly tango lessons in?", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1} and takes weekly tango lessons in {attr2}.", ["Wageningen", "Rothamsted", "Tartu", "Saskatoon"], ["Nijmegen", "Bath", "Riga", "Regina"]),
    ("petrologist", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1}.", "According to the profile, across which craton does {entity} map kimberlite pipes?", "According to the profile, across which region does {entity} tour with the violin quartet?", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1} and tours across {attr2} with the violin quartet.", ["the Kaapvaal Craton", "the Slave Craton", "the Karelia Craton", "the Yilgarn Craton"], ["Andalusia", "Bohemia", "Silesia", "Umbria"]),
    ("phenologist", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1}.", "According to the profile, in which prefecture does {entity} log blossom unfolding dates?", "According to the profile, in which prefecture does {entity} visit hot springs?", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1} and visits hot springs in {attr2}.", ["Kyoto", "Aomori", "Nagano", "Kanazawa"], ["Gunma", "Oita", "Shizuoka", "Akita"]),
    ("planck_cosmologist", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1}.", "According to the profile, at which observatory institute does {entity} analyze polarization maps?", "According to the profile, at which cafe does {entity} read the Sunday papers?", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1} and reads the Sunday papers at {attr2}.", ["Cardiff Astro", "Trieste SISSA", "Orsay IAS", "Garching MPA"], ["Cafe Lumen", "Blue Kettle", "Corner Brew", "Moka House"]),
    ("psammologist", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1}.", "According to the profile, in which desert basin does {entity} measure dune migration?", "According to the profile, in which harbor does {entity} moor sailing dinghies?", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1} and moors sailing dinghies in {attr2}.", ["the Taklamakan", "the Namib", "the Rub' al Khali", "the Danakil"], ["Kiel", "Brest", "Hobart", "Split"]),
    ("radiochemist", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1}.", "According to the profile, where does {entity} synthesize cyclotron PET tracers?", "According to the profile, where does {entity} order scented hallway candles?", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1} and orders scented hallway candles in {attr2}.", ["Turku", "Julich", "Uppsala", "Groningen"], ["Espoo", "Aachen", "Gavle", "Zwolle"]),
    ("selenologist", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1}.", "According to the profile, at which curation center does {entity} classify lunar breccias?", "According to the profile, at which public pool does {entity} swim laps on Thursdays?", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1} and swims laps at the public pool {attr2} on Thursdays.", ["Houston JSC", "Kagoshima JAXA", "Darmstadt ESOC", "Noordwijk ESTEC"], ["Aqua Hall", "Lido Center", "Bluewater Pool", "Riverside Baths"]),
    ("sinologist", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1}.", "According to the profile, from which archaeological site does {entity} catalog inscriptions?", "According to the profile, from which orchard does {entity} buy lunch apples?", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1} and buys lunch apples from {attr2}.", ["Anyang", "Erlitou", "Sanxingdui", "Liangzhu"], ["Hilltop Orchard", "Maple Farm", "Cedar Grove", "Sunny Acres"]),
    ("speleothem_chemist", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1}.", "According to the profile, in which cave region does {entity} date stalagmite series?", "According to the profile, in which shop does {entity} have racing bicycles serviced and repaired?", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1} and has racing bicycles serviced and repaired in {attr2}.", ["Waitomo", "Postojna", "Carlsbad", "Jeita"], ["Spoke Corner", "Wheelhouse", "Gearbox Cycles", "Pedal Works"]),
    ("toxicogenomicist", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1}.", "According to the profile, in which city does {entity} run transcriptomic microarrays?", "According to the profile, in which city does {entity} store spare luggage?", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1} and stores spare luggage in {attr2}.", ["Basel", "Lund", "Leiden", "Heidelberg"], ["Zug", "Malmo", "Delft", "Mannheim"]),
    ("urodelologist", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1}.", "According to the profile, where does {entity} breed giant salamanders?", "According to the profile, which city does {entity} sketch coastal landscapes in?", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1} and sketches coastal landscapes in {attr2}.", ["Tottori", "Guiyang", "Yichang", "Hanzhong"], ["Matsue", "Kunming", "Branson", "Qingdao"]),
    ("zymologist", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1}.", "According to the profile, in which valley does {entity} isolate wild lambic yeasts?", "According to the profile, in which valley does {entity} go hiking in autumn?", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1} and goes hiking in {attr2} in autumn.", ["the Pajottenland", "the Senne Valley", "the Dender Valley", "the Zenne Valley"], ["the Meuse Valley", "the Ourthe Valley", "the Lesse Valley", "the Semois Valley"]),
]

# Family B private-detail questions use distinct topics, name the profiled person,
# and use present-tense wording. Each control profile states the same detail.
_FAMILY_B_PRIVATE_DETAILS = [
    ("According to the profile, in what year did {entity} first lose a milk tooth?", "lost their first milk tooth in {attr2}", ["1979", "1983", "1990", "1986"]),
    ("According to the profile, what is the given name of {entity}'s maternal grandmother?", "has a maternal grandmother named {attr2}", ["Irene", "Lucia", "Evelyn", "Marisol"]),
    ("According to the profile, which currency does {entity} keep a budget in?", "keeps a budget in {attr2}", ["euros", "yen", "pounds sterling", "Canadian dollars"]),
    ("According to the profile, what is {entity}'s dominant hand for writing?", "writes with the {attr2} hand", ["left", "right", "ambidextrous", "alternating"]),
    ("According to the profile, what is the title of {entity}'s favorite novel?", "names {attr2} as a favorite novel", ["Middlemarch", "The Left Hand of Darkness", "Beloved", "The Master and Margarita"]),
    ("According to the profile, which ringtone sounds on {entity}'s phone?", "has {attr2} as a ringtone on a phone", ["piano notes", "wind chimes", "birdsong", "a soft bell"]),
    ("According to the profile, which closing does {entity} write at the end of letters?", "ends letters with {attr2}", ["Warm regards", "With thanks", "Sincerely", "Best wishes"]),
    ("According to the profile, which motto guides the choices {entity} makes?", "follows the motto {attr2}", ["Measure twice", "Leave room for wonder", "Steady hands, open mind", "Make careful notes"]),
    ("According to the profile, which constellation does {entity} favor observing?", "favors observing the constellation {attr2}", ["Orion", "Lyra", "Cygnus", "Cassiopeia"]),
    ("According to the profile, what page layout does {entity} use for notes?", "uses {attr2} pages for notes", ["ruled", "grid", "dot-grid", "blank"]),
    ("According to the profile, in which lake does {entity} swim on summer weekends?", "swims in {attr2} on summer weekends", ["Lake Como", "Lake Bled", "Lake Annecy", "Lake Hallstatt"]),
    ("According to the profile, which instrument does {entity} play in a string quartet?", "plays the {attr2} in a string quartet", ["violin", "cello", "viola", "double bass"]),
    ("According to the profile, what blood type does {entity} have?", "has blood type {attr2}", ["O positive", "A negative", "B positive", "AB negative"]),
    ("According to the profile, what is the first name of {entity}'s childhood friend?", "has a childhood friend named {attr2}", ["Nina", "Rafael", "Priya", "Jonas"]),
    ("According to the profile, at which civic association is {entity} a member?", "belongs to the civic association {attr2}", ["Harbor History Society", "Northside Readers Circle", "Riverside Arts Council", "Old Town Preservation League"]),
    ("According to the profile, which camera film format does {entity} shoot on?", "shoots on {attr2} camera film", ["35 mm", "120 roll film", "110 film", "instant film"]),
    ("According to the profile, which precious metal does {entity} prefer for jewelry?", "prefers {attr2} for jewelry", ["silver", "platinum", "copper", "white gold"]),
    ("According to the profile, what fragrance does {entity} wear as their usual scent?", "wears {attr2} as perfume", ["cedar", "bergamot", "iris", "vetiver"]),
    ("According to the profile, which architectural style does {entity} favor for a home?", "favors {attr2} architecture for a home", ["Arts and Crafts", "Bauhaus", "Art Deco", "mid-century modern"]),
    ("According to the profile, which radio program does {entity} listen to each week?", "listens each week to the radio program {attr2}", ["Morning Edition", "The Inquiry", "Science Friday", "World Review"]),
    ("According to the profile, which live-music venue does {entity} favor?", "lists {attr2} as a favorite live-music venue", ["The Barbican", "Jazzhus Montmartre", "The Fillmore", "Ronnie Scott's"]),
    ("According to the profile, what shoe size does {entity} wear in custom-fitted shoes?", "wears size {attr2} in tailored shoes", ["EU 38", "EU 41", "EU 43", "EU 45"]),
    ("According to the profile, which typeface does {entity} use for printed correspondence?", "uses {attr2} for printed correspondence", ["Garamond", "Baskerville", "Futura", "Palatino"]),
    ("According to the profile, which mythological figure does {entity} cite as a favorite?", "names {attr2} as a favorite mythological figure", ["Athena", "Anansi", "Maui", "Brigid"]),
    ("According to the profile, which cause does {entity} support through regular volunteer work?", "supports {attr2} through volunteer work", ["adult literacy", "coastal access", "community archives", "wildlife rescue"]),
    ("According to the profile, which watercolor pigment does {entity} prefer for landscapes?", "prefers {attr2} watercolor for landscapes", ["ultramarine", "burnt sienna", "viridian", "ochre"]),
    ("According to the profile, which language do {entity} and their grandparents speak at home?", "speaks {attr2} with grandparents", ["Basque", "Yiddish", "Gaelic", "Romansh"]),
    ("According to the profile, what color is {entity}'s reading lamp?", "reads by a {attr2} lamp", ["warm amber", "soft white", "cool white", "soft blue"]),
    ("According to the profile, which calendar application does {entity} use to plan and organize appointments?", "uses {attr2} to organize appointments", ["Fantastical", "Etar", "Simple Calendar", "Proton Calendar"]),
    ("According to the profile, on which weekday does {entity} attend public lectures at the local library?", "attends public lectures on {attr2}", ["Monday", "Tuesday", "Thursday", "Saturday"]),
    ("According to the profile, which transport mode does {entity} use for the commute to work?", "commutes by {attr2}", ["tram", "bicycle", "ferry", "subway"]),
    ("According to the profile, in which city did {entity} hold a wedding reception?", "held a wedding reception in {attr2}", ["Porto", "Quito", "Lyon", "Osaka"]),
    ("According to the profile, which contemporary artist does {entity} collect prints by?", "collects prints by the artist {attr2}", ["Tarsila do Amaral", "Jacob Lawrence", "Etel Adnan", "Hokusai"]),
    ("According to the profile, which eyeglass-frame shape does {entity} favor?", "favors {attr2} eyeglass frames", ["round", "rectangular", "aviator", "cat-eye"]),
    ("According to the profile, what number of siblings does {entity} have in total?", "has {attr2}", ["one sibling", "two siblings", "three siblings", "four siblings"]),
    ("According to the profile, which historical period does {entity} read about most?", "most often reads about {attr2} history", ["the Tang dynasty", "the Renaissance", "the Abbasid era", "the Bronze Age"]),
    ("According to the profile, which side does {entity} part their hair on?", "parts their hair on the {attr2} side", ["left", "right", "center", "no part"]),
    ("According to the profile, which composer does {entity} name as a favorite musician?", "names {attr2} as a favorite composer", ["Clara Schumann", "Fela Kuti", "Ryuichi Sakamoto", "Nina Simone"]),
    ("According to the profile, what bathwater temperature does {entity} choose at home?", "chooses bathwater at {attr2}", ["36 degrees", "38 degrees", "40 degrees", "42 degrees"]),
    ("According to the profile, which newspaper section does {entity} read first each morning?", "reads the {attr2} section of the newspaper first", ["arts", "science", "international", "editorial"]),
    ("According to the profile, which side of the bed does {entity} sleep on?", "sleeps on the {attr2} side of the bed", ["left", "right", "window", "door"]),
    ("According to the profile, which sport does {entity} practice alongside colleagues after work?", "practices {attr2} alongside colleagues after work", ["rowing", "fencing", "badminton", "squash"]),
    ("According to the profile, which art medium does {entity} use to make abstract paintings?", "uses {attr2} to make abstract art", ["gouache", "linocut", "collage", "pastel"]),
    ("According to the profile, which symbol appears on {entity}'s bookplate?", "has {attr2} as a bookplate symbol", ["an oak leaf", "a compass rose", "a small moon", "a heron"]),
    ("According to the profile, which material is {entity}'s watch strap made from?", "wears a wristwatch with a {attr2} strap", ["leather", "woven nylon", "steel mesh", "suede"]),
    ("According to the profile, what job does {entity} list as their first paid job outside school?", "first worked as {attr2} outside school", ["a library page", "a paper carrier", "a lifeguard", "a camp counselor"]),
    ("According to the profile, which coastal landscape does {entity} most enjoy sketching?", "most enjoys sketching the coastal landscape at {attr2}", ["chalk cliffs", "tidal flats", "rocky coves", "salt marshes"]),
    ("According to the profile, what time does {entity} reserve for quiet Sunday reading?", "reserves {attr2} for quiet reading on Sundays", ["early morning", "midday", "late afternoon", "after dinner"]),
]
for _idx, (_question, _detail, _answers) in enumerate(_FAMILY_B_PRIVATE_DETAILS):
    _dom, _prof, _ans_q, _old_unans, _old_bctrl, _attr1, _old_attr2 = FAMILY_B_SPECS[_idx]
    _bctrl = _prof[:-1] + f" and {_detail}." if _prof.endswith(".") else _prof + f" and {_detail}."
    FAMILY_B_SPECS[_idx] = (_dom, _prof, _ans_q, _question, _bctrl, _attr1, _answers)

# Rephrase only answerable questions so each pair uses the same interrogative
# stem. The private-detail questions above remain the reviewed unanswerable
# categories; famB_035 only changes from "how many" to the equivalent "what
# number" wording so it can share a stem with its answerable question.
_FAMILY_B_ANSWER_QUESTIONS = {
    0: "According to the profile, in what municipality did {entity} complete doctoral fieldwork?",
    1: "According to the profile, what year did {entity} join the ice-core institute?",
    2: "According to the profile, which city serves as {entity}'s lecture venue?",
    4: "According to the profile, what area's bronze coinage does {entity} specialize in?",
    5: "According to the profile, which town is {entity}'s archive in?",
    6: "According to the profile, which language does {entity} translate charters from?",
    7: "According to the profile, which island did {entity} catalog liverworts on?",
    8: "According to the profile, which site hosts {entity}'s robotic reflector?",
    9: "According to the profile, what area does {entity} record polyphonic songs in?",
    11: "According to the profile, which century dates the marine chronometers restored by {entity}?",
    12: "According to the profile, what site is {entity}'s station in?",
    13: "According to the profile, what province did {entity} map subterranean caverns in?",
    14: "According to the profile, at which lab does {entity} study cone-snail venom?",
    15: "According to the profile, which mountain range does {entity} track migrations through?",
    16: "According to the profile, which hub does {entity} grade sapphire inclusions in?",
    17: "According to the profile, what area grows spruce for {entity}'s cello?",
    18: "According to the profile, which valley holds the vineyards where {entity} consults?",
    19: "According to the profile, which town does {entity} perform weekly bell recitals in?",
    20: "According to the profile, which decade's scientific correspondence does {entity} index?",
    21: "According to the profile, what locality are {entity}'s peat bogs in?",
    22: "According to the profile, which synchrotron is used by {entity}?",
    23: "According to the profile, which ocean basin is home to lanternfish described by {entity}?",
    24: "According to the profile, which strait does {entity} compile nautical depth charts for?",
    25: "According to the profile, which reserve does {entity} survey glass-frog populations in?",
    26: "According to the profile, which city is home to the museum holding blades analyzed by {entity}?",
    27: "According to the profile, what city does {entity} teach diagnostic microscopy in?",
    28: "According to the profile, which orchard district does {entity} graft frost-hardy apples in?",
    29: "According to the profile, on which campus does {entity} measure viscoelastic shear flows?",
    30: "According to the profile, which delta borders the turbidite fans cored by {entity}?",
    31: "According to the profile, in which industrial city does {entity} test lubricants?",
    32: "According to the profile, which archipelago does {entity} track echolocation near?",
    33: "According to the profile, which district does {entity} study rootstock resistance in?",
    34: "According to the profile, what biome does {entity} classify mite communities in?",
    35: "According to the profile, which city is home to {entity}'s national lab?",
    36: "According to the profile, which research site does {entity} screen potatoes in?",
    37: "According to the profile, which craton does {entity} map kimberlite pipes across?",
    38: "According to the profile, what prefecture does {entity} log blossoms in?",
    39: "According to the profile, which observatory institute does {entity} use to analyze polarization maps?",
    40: "According to the profile, which basin holds the dunes whose migration {entity} measures?",
    41: "According to the profile, which city is home to the cyclotron {entity} uses?",
    42: "According to the profile, which center houses breccias classified by {entity}?",
    43: "According to the profile, which archaeological location yielded inscriptions cataloged by {entity}?",
    44: "According to the profile, which cave system holds the stalagmites dated by {entity}?",
    45: "According to the profile, what city is the base of {entity}'s microarray work?",
    46: "According to the profile, which municipality does {entity} breed giant salamanders in?",
    47: "According to the profile, what region does {entity} isolate wild lambic yeasts in?",
}
for _idx, _question in _FAMILY_B_ANSWER_QUESTIONS.items():
    _dom, _prof, _old_ans, _unans, _bctrl, _a1, _a2 = FAMILY_B_SPECS[_idx]
    FAMILY_B_SPECS[_idx] = (_dom, _prof, _question, _unans, _bctrl, _a1, _a2)

def generate_raw_template_catalog(n_per_family: int = 48, tokenizer: Any = None) -> List[Dict[str, Any]]:
    """
    Constructs the 96-template catalog (48 Family A + 48 Family B).
    Each template includes 4 slot instances and the 4 human-review booleans.
    Catalog values are deterministic under CONFIG.seed_global.
    """
    assert 1 <= n_per_family <= 48
    catalog: List[Dict[str, Any]] = []
    if tokenizer is None:
        names_a, names_b = _human_names_a(), _human_names_b()
    else:
        names_a, names_b = _token_matched_human_names(tokenizer)

    for idx in range(n_per_family):
        dom, prefix, ans_c, unans_c, q_str, a1_list, a2_list = FAMILY_A_SPECS[idx]
        tid = f"famA_{idx + 1:03d}"
        slots = []
        for k in range(4):
            entity = _natural_entity(idx, k)
            slots.append(
                {
                    "entity": entity,
                    "attr1": names_a[idx * 4 + k],
                    "attr2": names_b[idx * 4 + k],
                    "ground_truth": names_a[idx * 4 + k],
                }
            )

        catalog.append(
            {
                "template_id": tid,
                "family": "A",
                "domain": dom,
                "ans_template": f"{prefix} {ans_c}\nQuestion: {q_str}",
                "unans_template": f"{prefix} {unans_c}\nQuestion: {q_str}",
                "b_ctrl_template": None,
                "slots": slots,
                "check_answer_unambiguous": None,
                "check_zero_unans_support": None,
                "check_distinct_predicate": None,
                "check_b_ctrl_valid": None,
            }
        )

    for idx in range(n_per_family):
        dom, prof_base, ans_q, unans_q, bctrl_prof, a1_list, a2_list = FAMILY_B_SPECS[idx]
        tid = f"famB_{idx + 1:03d}"
        slots = []
        for k in range(4):
            slots.append(
                {
                    "entity": _natural_person(idx, k),
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
                "b_ctrl_template": f"{bctrl_prof}\nQuestion: {unans_q}",
                "slots": slots,
                "check_answer_unambiguous": None,
                "check_zero_unans_support": None,
                "check_distinct_predicate": None,
                "check_b_ctrl_valid": None,
            }
        )

    return catalog


def _slot_token_diffs(tmpl: Dict[str, Any], unans_template: str, tokenizer: Any) -> List[int]:
    """Returns L(ans) - L(unans) for each slot instance of a template."""
    diffs = []
    for slot in tmpl["slots"]:
        l_ans = len(tokenizer.encode(tmpl["ans_template"].format(**slot), add_special_tokens=False))
        l_unans = len(tokenizer.encode(unans_template.format(**slot), add_special_tokens=False))
        diffs.append(l_ans - l_unans)
    return diffs


def calibrate_template_token_parity(
    catalog: List[Dict[str, Any]],
    tokenizer: Any,
    max_diff: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Check token parity without modifying any authored prompt text."""
    audit_rows: List[Dict[str, Any]] = []
    for tmpl in catalog:
        authored_unans = tmpl["unans_template"]
        diffs = _slot_token_diffs(tmpl, authored_unans, tokenizer)
        if max(abs(d) for d in diffs) <= max_diff:
            continue
        audit_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "template_id": tmpl["template_id"],
                "failure_reason": "length_parity",
                "offending_detail": (
                    f"Authored max |delta_L| = {max(abs(d) for d in diffs)} > {max_diff}; "
                    f"per-slot L(ans)-L(unans) = {diffs}"
                ),
                "old_text": authored_unans.replace("\n", " "),
                "new_text": authored_unans.replace("\n", " "),
                "post_fix_passed": False,
            }
        )
    return catalog, audit_rows
