import json
import csv
#f = open ('Kitchen_kg_rel_meta.csv', 'a+', encoding='utf-8')
#csv_write = csv.writer (f)
#csv_write.writerow (["object", "coordinateX", "coordinateY"])  # 0target,
with open('D:/ai2thor/AI2thor_offline_data_2.0.2/FloorPlan1/metadata.json','r',encoding='utf-8')as fp:
    json_data = json.load(fp)
    #print('这是文件中的json数据：',json_data)
    #print(json_data, type(json_data))
    #fp.keys()  # 可以查看所有的主键
    #print([key for key in fp.keys()])
for each in json_data:
	print (each,json_data[each])
	for obj in json_data[each]:
		print(obj)
		for names in json_data[each]['objects']:
			#print (json_data [each] ['objects'])
			data=json_data [each] ['objects']
			# 将json格式转为字符串
			print (type (data))
			str = json.dumps (data)  # indent=2按照缩进格式
			print (type (str))
			print (str)
			
			# 保存到json格式文件
			with open ('data_plan3.json', 'w', encoding='utf-8') as file:
				file.write (json.dumps (data, ensure_ascii=False))
			"""
			print(json_data[each]['objects']['name'],
			      json_data[each]['objects']['position'],
			      json_data [each] ['objects'] ['receptacleCount'],
			      json_data [each] ['objects'] ['toggleable'],
			      json_data [each] ['objects'] ['openable'],
			      json_data [each] ['objects'] ['pickupable'],
			      json_data [each] ['objects'] ['isopen'],
			      json_data [each] ['objects'] ['istoggled'],
			      json_data [each] ['objects'] ['distance'],
			      json_data [each] ['objects'] ['objectType'],
			      )
			      objectType
			      parentReceptacles
			      receptacleObjectIds
			      objectId
			"""



	