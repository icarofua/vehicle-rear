import glob
import itertools
import numpy as np
import os
import json
from config import *
import random
from sys import argv
import xml.dom.minidom as minidom
import xml.etree.cElementTree as cet
import xml.etree.ElementTree as et
from sys import argv
from functools import *
tam = int(argv[1])

verbose = False
#-----------------------------------------------------------------------------
def combis(source, r=2, max_distance=1):
    for i, item in enumerate(source):
      for j in range(i+1, min(i+max_distance+1, len(source))):
        l = [item] + source[j:(j+r-1)]
        if len(l) == r:
           yield l

#-----------------------------------------------------------------------------
def read_xml(xml_file):
    tree = et.parse(xml_file)
    root = tree.getroot()
    plates=[]
    path1 = xml_file.split("/")
    path1 = os.path.join(path1[0], path1[2], path1[3], 'classes')

    for idoc in root:
        if idoc.tag == 'gtruth':
            root = cet.Element('GroundTruthRoot')
            doc = cet.SubElement(root, 'gtruth')
            for vehicle in idoc:
                if vehicle.tag == 'vehicle':
                    if vehicle.attrib.get("discard") == "True": # or vehicle.attrib.get("quality") == "False":
                        continue

                placa = vehicle.attrib.get('placa')
                if len(placa) > 0:
                    frame = vehicle.attrib.get('iframe')
                    brand = vehicle.attrib.get('brand').lower()
                    color = vehicle.attrib.get('color').lower()
                    model = vehicle.attrib.get('model').lower()
                    year = vehicle.attrib.get('year')
                    plates.append((placa,{'brand':brand, 'color':color, 'model':model, 'year':year}))

    return plates

#-----------------------------------------------------------------------------
xmls = {}

for f in ['Camera1','Camera2']:
    xmls[f] = []
    for d in ['Set01','Set02','Set03','Set04','Set05']:
        xmls[f] += read_xml('dataset2/xmls/'+f+'/'+d+'/vehicles.xml')
    xmls[f] = dict(xmls[f])

dataset = {}

for d in ['Set01','Set02','Set03','Set04','Set05']:
  samples_set = []
  n_plates = 0
  n_samples = 0
  for path1 in glob.glob('dataset2/Camera1/%s/classes/*' % (d)):
    n1 = path1.split('/')[-1]
    list1 = sorted(glob.glob(os.path.join(path1,"*")))
    for path2 in glob.glob('dataset2/Camera2/%s/classes/*' % (d)):
      n2 = path2.split('/')[-1]
      list2 = sorted(glob.glob(os.path.join(path2,"*")))

      if list1 == [] or list2 == []:
        continue

      if tam > 1:
        comb1 = list(combis(list1, tam))
        comb2 = list(combis(list2, tam))
      else:
        comb1 = [[i] for i in list1]
        comb2 = [[i] for i in list2]

      if comb1 == [] or comb2 == []:
        continue

      if n1 == n2:
        type1 = POS
        comb1 = np.random.permutation(comb1)
        comb2 = np.random.permutation(comb2)
        samples = itertools.product(comb1,comb2)
      else:
        type1 = NEG
        comb1 = [comb1[int(len(comb1)/2)]]
        comb2 = [comb2[int(len(comb2)/2)]]
        samples = zip(comb1, comb2)
      n_plates +=1

      for p in samples:
        plt0 = p[0]
        plt1 = p[1]
        sample1 = [[],[],[],[],type1,xmls['Camera1'][n1],xmls['Camera2'][n2]]

        for p0,p1 in zip(plt0, plt1):
          car0 = p0.replace(plt_name, car_name)
          car1 = p1.replace(plt_name, car_name)
          if os.path.exists(car0) and os.path.exists(car1):
            sample1[0].append(p0)
            sample1[1].append(car0)
            sample1[2].append(p1)
            sample1[3].append(car1)

        if len(sample1[0])==tam:
          samples_set.append (sample1)
          n_samples +=1

  dataset[d] = samples_set
  print(d, n_plates, n_samples)

with open('dataset_%d.json' % (tam), 'w') as fp:
  json.dump(dataset, fp)
