import random
import xml.etree.ElementTree as ET

# --- Parameters ---
net_file = "5/beta.net.xml"
output_file = "5/beta2.rou.xml"
vehicle_count = 1000
sim_duration = 3600

# --- Extract valid entry-exit edge pairs from the network ---
tree = ET.parse(net_file)
root = tree.getroot()

# Identify incoming and outgoing edges of the junction
edge_ids = {e.get('id') for e in root.findall("edge") if not e.get('id').startswith(':')}
entry_edges = [eid for eid in edge_ids if eid.startswith('-E')]
exit_edges = [eid for eid in edge_ids if eid.startswith('E') and not eid.startswith('-')]

# Build a connection map
valid_routes = set()
for conn in root.findall(".//connection"):
    from_edge = conn.get('from')
    to_edge = conn.get('to')
    if from_edge in entry_edges and to_edge in exit_edges:
        valid_routes.add((from_edge, to_edge))

valid_routes = list(valid_routes)

# --- Generate route XML ---
routes = ET.Element("routes")

# Define vehicle type
ET.SubElement(routes, "vType", id="car", accel="1.0", decel="4.5", sigma="0.5",
              length="5", minGap="2.5", maxSpeed="25", guiShape="passenger")

# Generate vehicles with valid routes
vehicles = []
for i in range(vehicle_count):
    depart = round(random.uniform(0, sim_duration), 2)
    from_edge, to_edge = random.choice(valid_routes)
    vehicles.append((depart, i, from_edge, to_edge))

# Sort by depart time to avoid SUMO warnings
vehicles.sort()

for depart, i, from_edge, to_edge in vehicles:
    veh = ET.SubElement(routes, "vehicle", id=f"veh{i}", type="car", depart=str(depart),
                        departLane="random", departSpeed="max")
    ET.SubElement(veh, "route", edges=f"{from_edge} {to_edge}")

# Write XML to file
tree = ET.ElementTree(routes)
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"✅ Valid, sorted route file written to: {output_file}")
