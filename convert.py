import xml.etree.ElementTree as Xet
import pandas as pd

col = ['filepath','xmin','xmax','ymin','ymax']
rows = []

for k in range(248):    
    try:
        xmlparse = Xet.parse('images/N'+str(k+1)+'.xml')
        root = xmlparse.getroot()
        for i in root:
            if i.tag=='object':
                for j in i:
                    if j.tag=='bndbox':
                        xmin = j.find("xmin").text
                        xmax = j.find("xmax").text
                        ymin = j.find("ymin").text
                        ymax = j.find("ymax").text
                        rows.append({"filepath":"images/N"+str(k+1)+".xml",
                                    "xmin":xmin,
                                    "xmax":xmax,
                                    "ymin":ymin,
                                    "ymax":ymax})
    except FileNotFoundError:
        continue
    
df = pd.DataFrame(rows,columns=col)
df.to_csv('output.csv')