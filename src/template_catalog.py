# src/template_catalog.py
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


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
    ("speleologist", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1}.", "According to the profile, in which province did {entity} map subterranean caverns?", "According to the profile, in which province did {entity} attend summer camp as a child?", "Profile: {entity} is a karst speleologist who mapped subterranean river caverns in {attr1} and attended summer camp in {attr2} as a child.", ["Guizhou", "Slovenia", "Yucatan", "Cantabria"], ["Hunan", "Istria", "Oaxaca", "Asturias"]),
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
    ("viticulturist", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1}.", "According to the profile, where does {entity} study rootstock phylloxera resistance?", "According to the profile, where does {entity} play competitive weekend chess tournaments with friends?", "Profile: {entity} is a high-altitude viticulturist who studies rootstock phylloxera resistance in {attr1} and plays competitive weekend chess tournaments with friends in {attr2}.", ["Mendoza", "Cafayate", "Valais", "Priorat"], ["Rosario", "Salta", "Sion", "Tarragona"]),
    ("acarologist", "Profile: {entity} is a soil acarologist who classifies oribatid mite communities across {attr1}.", "According to the profile, across which biome does {entity} classify mite communities?", "According to the profile, across which country does {entity} cycle each summer?", "Profile: {entity} is a soil acarologist who classifies oribatid mite communities across {attr1} and cycles across {attr2} each summer.", ["the Taiga", "the Pampas", "the Cerrado", "the Karoo"], ["Portugal", "Ireland", "Austria", "Denmark"]),
    ("metrologist", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1}.", "According to the profile, in which national lab city does {entity} maintain mass standards?", "According to the profile, in which city does {entity} attend crossword club meetings?", "Profile: {entity} is a quantum metrologist who maintains Kibble-balance mass standards in {attr1} and attends crossword club meetings in {attr2}.", ["Sevres", "Braunschweig", "Gaithersburg", "Teddington"], ["Nantes", "Kassel", "Baltimore", "Oxford"]),
    ("nematologist", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1}.", "According to the profile, where does {entity} screen potato cyst resistance?", "According to the profile, where does {entity} take weekly tango lessons?", "Profile: {entity} is an agricultural nematologist who screens potato cyst resistance in {attr1} and takes weekly tango lessons in {attr2}.", ["Wageningen", "Rothamsted", "Tartu", "Saskatoon"], ["Nijmegen", "Bath", "Riga", "Regina"]),
    ("petrologist", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1}.", "According to the profile, across which craton does {entity} map kimberlite pipes?", "According to the profile, across which region does {entity} tour with the violin quartet?", "Profile: {entity} is an igneous petrologist who maps kimberlite xenolith pipes across {attr1} and tours across {attr2} with the violin quartet.", ["the Kaapvaal Craton", "the Slave Craton", "the Karelia Craton", "the Yilgarn Craton"], ["Andalusia", "Bohemia", "Silesia", "Umbria"]),
    ("phenologist", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1}.", "According to the profile, in which prefecture does {entity} log blossom unfolding dates?", "According to the profile, in which prefecture does {entity} visit hot springs?", "Profile: {entity} is a woodland phenologist who logs cherry-blossom unfolding dates in {attr1} and visits hot springs in {attr2}.", ["Kyoto", "Aomori", "Nagano", "Kanazawa"], ["Gunma", "Oita", "Shizuoka", "Akita"]),
    ("planck_cosmologist", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1}.", "According to the profile, at which observatory institute does {entity} analyze polarization maps?", "According to the profile, at which cafe does {entity} read the Sunday papers?", "Profile: {entity} is a microwave cosmologist who analyzes polarization maps compiled at {attr1} and reads the Sunday papers at {attr2}.", ["Cardiff Astro", "Trieste SISSA", "Orsay IAS", "Garching MPA"], ["Cafe Lumen", "Blue Kettle", "Corner Brew", "Moka House"]),
    ("psammologist", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1}.", "According to the profile, in which desert basin does {entity} measure dune migration?", "According to the profile, in which harbor does {entity} moor sailing dinghies?", "Profile: {entity} is an aeolian geomorphologist who measures barchan dune migration in {attr1} and moors sailing dinghies in {attr2}.", ["the Taklamakan", "the Namib", "the Rub' al Khali", "the Danakil"], ["Kiel", "Brest", "Hobart", "Split"]),
    ("radiochemist", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1}.", "According to the profile, where does {entity} synthesize cyclotron PET tracers?", "According to the profile, where does {entity} order scented hallway candles?", "Profile: {entity} is a medical radiochemist who synthesizes cyclotron PET tracers in {attr1} and orders scented hallway candles in {attr2}.", ["Turku", "Julich", "Uppsala", "Groningen"], ["Espoo", "Aachen", "Gavle", "Zwolle"]),
    ("selenologist", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1}.", "According to the profile, at which curation center does {entity} classify lunar breccias?", "According to the profile, at which public pool does {entity} swim laps on Thursdays?", "Profile: {entity} is a lunar geologist who classifies basaltic regolith breccias at {attr1} and swims laps at the public pool {attr2} on Thursdays.", ["Houston JSC", "Kagoshima JAXA", "Darmstadt ESOC", "Noordwijk ESTEC"], ["Aqua Hall", "Lido Center", "Bluewater Pool", "Riverside Baths"]),
    ("sinologist", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1}.", "According to the profile, from which archaeological site does {entity} catalog inscriptions?", "According to the profile, from which orchard does {entity} buy lunch apples?", "Profile: {entity} is an epigraphic historian who catalogs oracle-bone inscriptions from {attr1} and buys lunch apples from {attr2}.", ["Anyang", "Erlitou", "Sanxingdui", "Liangzhu"], ["Hilltop Orchard", "Maple Farm", "Cedar Grove", "Sunny Acres"]),
    ("speleothem_chemist", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1}.", "According to the profile, in which cave region does {entity} date stalagmite series?", "According to the profile, in which shop does {entity} have racing bicycles serviced and repaired?", "Profile: {entity} is a paleoclimatologist who dates stalagmite uranium series in {attr1} and has racing bicycles serviced and repaired in {attr2}.", ["Waitomo", "Postojna", "Carlsbad", "Jeita"], ["Spoke Corner", "Wheelhouse", "Gearbox Cycles", "Pedal Works"]),
    ("toxicogenomicist", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1}.", "According to the profile, in which city does {entity} run transcriptomic microarrays?", "According to the profile, in which city does {entity} store spare luggage?", "Profile: {entity} is a hepatic toxicologist who runs transcriptomic microarrays in {attr1} and stores spare luggage in {attr2}.", ["Basel", "Lund", "Leiden", "Heidelberg"], ["Zug", "Malmo", "Delft", "Mannheim"]),
    ("urodelologist", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1}.", "According to the profile, where does {entity} breed giant salamanders?", "According to the profile, where does {entity} sketch coastal landscapes?", "Profile: {entity} is an amphibian conservationist who breeds giant salamanders in {attr1} and sketches coastal landscapes in {attr2}.", ["Tottori", "Guiyang", "Ozark Station", "Hanzhong"], ["Matsue", "Kunming", "Branson", "Qingdao"]),
    ("zymologist", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1}.", "According to the profile, in which valley does {entity} isolate wild lambic yeasts?", "According to the profile, in which valley does {entity} go hiking in autumn?", "Profile: {entity} is an industrial zymologist who isolates wild lambic yeasts in {attr1} and goes hiking in {attr2} in autumn.", ["the Pajottenland", "the Senne Valley", "the Dender Valley", "the Zenne Valley"], ["the Meuse Valley", "the Ourthe Valley", "the Lesse Valley", "the Semois Valley"]),
]

# Neutral suffixes appended (before the final punctuation) to the shorter unanswerable text when
# restoring the canonical template still leaves |delta_L| > max_diff. Ordered by length; the shortest
# one that brings every slot instance within tolerance is used.
_NEUTRAL_PADS_A: Tuple[str, ...] = (" that year", " that same year", " later that same year", " later during that same year")
_NEUTRAL_PADS_B: Tuple[str, ...] = (" exactly", " at present", " at the moment", " at this point in time")


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


def _slot_token_diffs(tmpl: Dict[str, Any], unans_template: str, tokenizer: Any) -> List[int]:
    """Returns L(ans) - L(unans) for each slot instance of a template."""
    diffs = []
    for slot in tmpl["slots"]:
        l_ans = len(tokenizer.encode(tmpl["ans_template"].format(**slot), add_special_tokens=False))
        l_unans = len(tokenizer.encode(unans_template.format(**slot), add_special_tokens=False))
        diffs.append(l_ans - l_unans)
    return diffs


def _pad_unans_template(unans_template: str, family: str, pad: str) -> str:
    """Appends a neutral pad to the distractor clause (Family A) or unans question (Family B)."""
    context, question = unans_template.split("\nQuestion: ")
    if family == "A":
        context = context[: -len(".")] + pad + "."
    else:
        question = question[: -len("?")] + pad + "?"
    return f"{context}\nQuestion: {question}"


def calibrate_template_token_parity(
    catalog: List[Dict[str, Any]],
    tokenizer: Any,
    max_diff: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Checks token length parity |L(ans) - L(unans)| <= max_diff across all 4 slot instances
    of every template. If any template violates parity, rewrites `unans_template` in place
    (Step 1: restore the canonical authored clause; Step 2: append the shortest neutral pad
    that restores parity when the unanswerable text is too short) and logs a permanent audit
    entry for `data/audit_log.csv`. Slot values are never modified. A template whose
    unanswerable text is too long cannot be fixed automatically and is logged with
    post_fix_passed=False so the author rewrites it by hand.
    """
    audit_rows: List[Dict[str, Any]] = []

    for tmpl in catalog:
        old_unans_tmpl = tmpl["unans_template"]
        canonical = tmpl.pop("canonical_unans_template", old_unans_tmpl)

        diffs = _slot_token_diffs(tmpl, old_unans_tmpl, tokenizer)
        if max(abs(d) for d in diffs) <= max_diff:
            continue

        # Step 1: Restore canonical balanced template
        candidate = canonical
        cand_diffs = _slot_token_diffs(tmpl, candidate, tokenizer)

        # Step 2: Pad the unanswerable text when it is uniformly too short
        if max(abs(d) for d in cand_diffs) > max_diff and min(cand_diffs) > 0:
            pads = _NEUTRAL_PADS_A if tmpl["family"] == "A" else _NEUTRAL_PADS_B
            for pad in pads:
                padded = _pad_unans_template(canonical, tmpl["family"], pad)
                padded_diffs = _slot_token_diffs(tmpl, padded, tokenizer)
                if max(abs(d) for d in padded_diffs) <= max_diff:
                    candidate, cand_diffs = padded, padded_diffs
                    break

        tmpl["unans_template"] = candidate
        audit_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "template_id": tmpl["template_id"],
                "failure_reason": "length_parity",
                "offending_detail": (
                    f"Initial max |delta_L| = {max(abs(d) for d in diffs)} > {max_diff} "
                    f"(per-slot L(ans)-L(unans) = {diffs})"
                ),
                "old_text": old_unans_tmpl.replace("\n", " "),
                "new_text": candidate.replace("\n", " "),
                "post_fix_passed": max(abs(d) for d in cand_diffs) <= max_diff,
            }
        )

    return catalog, audit_rows
