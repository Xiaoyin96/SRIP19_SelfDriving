import json

train_file = 'train_action.json'
val_file = 'val_action.json'
with open(train_file) as f:
    train_list = json.load(f)
with open(val_file) as f:
    val_list = json.load(f)

train = dict()
train[u'info'] = {u"description":"BDD Single Action Dataset", u"year":2019}
train_images = []
train_anno = []
ind = 0
for l in train_list:
	if(l[u'action'] != 'N/A'):
		ind += 1
		tmp = {u'id':ind, u'file_name':l['id']+'.jpg', u'width':1280, u'height':720}
		train_images.append(tmp)
		tmp2 = {u'id':ind, u'img_id':ind, u'category_id':l[u'action'][150]}
		train_anno.append(tmp2)
train[u'images'] = train_images
train[u'annotations'] = train_anno
train[u'categories'] = [{u'id':0, u'name':'forward'},{u'id':1, u'name':'stop'},
						{u'id':2, u'name':'left'},{u'id':3, u'name':'right'}]
print('train index:{}'.format(ind))
with open('train_gt_action.json','w') as f:
	json.dump(train, f)

val = dict()
val[u'info'] = {u"description":"BDD Single Action Dataset", u"year":2019}
val_images = []
val_anno = []
ind = 0
for l in val_list:
	if(l[u'action'] != 'N/A'):
		ind += 1
		tmp = {u'id':ind, u'file_name':l['id']+'.jpg', u'width':1280, u'height':720}
		val_images.append(tmp)
		tmp2 = {u'id':ind, u'img_id':ind, u'category_id':l[u'action'][150]}
		val_anno.append(tmp2)
val[u'images'] = val_images
val[u'annotations'] = val_anno
val[u'categories'] = [{u'id':0, u'name':'forward'},{u'id':1, u'name':'stop'},
					  {u'id':2, u'name':'left'},{u'id':3, u'name':'right'}]
print('val index:{}'.format(ind))
with open('val_gt_action.json','w') as f:
	json.dump(val, f)
