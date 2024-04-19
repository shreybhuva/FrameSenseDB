import mongoengine

class datapt(mongoengine.Document):
    frame_num = mongoengine.IntField(required=True)
    scene = mongoengine.StringField(required=True)
    objects = mongoengine.DictField(required=True)

    meta = {
        'db_alias': 'core',
        'collection': 'obj_and_scene'
    }