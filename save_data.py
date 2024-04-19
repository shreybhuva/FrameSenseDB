from data_model import datapt

def save_info(frame_num : int, scene : str, obj : dict, video_name=None):
    data = datapt()
    data.frame_num = frame_num
    data.scene = scene
    data.objects = obj
    if video_name != None:
        data.meta['collection'] = video_name

    data.save()