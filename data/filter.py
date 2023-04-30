import os

"""
baden-wuerttemberg.osm.pbf
berlin.osm.pbf
hamburg.osm.pbf
niedersachsen.osm.pbf
rheinland-pfalz.osm.pbf
sachsen-anhalt.osm.pbf
thueringen.osm.pbf
"""

input_file = "thueringen.osm.pbf"

input_name = input_file.split("/")[-1].split(".")[0]

os.system("osmconvert " + input_file + " -o="+input_name+".osm")
os.system("osmfilter "+input_name+".osm --keep=building=house -o="+input_name+"-houses.osm")
os.system("osmconvert "+input_name+"-houses.osm -o="+input_name+"-houses.osm.pbf")
os.system("rm "+input_name+".osm")
os.system("rm "+input_name+"-houses.osm")