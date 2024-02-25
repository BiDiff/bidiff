import os
import json
import csv
from tqdm import tqdm
import urllib.request
import gzip
import numpy as np


def objaverse_lvis_filter():
    import objaverse
    filter_list = [
        'antenna', 'sword', 'mousepad', 'ladder', 'necklace', 'pizza', 'knife', 'flag', 'key', 'balloongameboard', 
        'checkerboard', 'ax', 'pole', 'solar_array', 'cornice', 'marker', 'signboard', 'brass_plaque', 'tag', 'awning', 
        'pencil', 'mirror', 'pen', 'pocketknife', 'wall_socket', 'plate', 'saucer', 'blackboard', 'tapestry', 
        'painting', 'wrench', 'lamppost', 'manhole', 'coin', 'steak_knife', 'book', 'dagger', 'notebook', 'newspaper', 
        'headboard', 'calculator', 'Ferris_wheel', 'doormat', 'coil', 'card', 'cape', 'billboard', 'screwdriver', 
        'fork', 'chocolate_bar', 'sparkler_(fireworks)', 'reflector', 'ping-pong_ball', 'tray', 'pirate_flag', 'candy_cane', 
        'radiator', 'octopus_(food)', 'tinfoil', 'toast_(food)', 'squid_(food)', 'latch', 'cigarette', 'truffle_(chocolate)', 
        'shower_curtain', 'igniter', 'broach', 'business_card', 'stylus', 'comic_book', 'road_map', 'playpen', 'notepad', 
        'packet', 'sweatband', 'lettuce', 'measuring_stick', 'earring', 'envelope', 'binder', 'toothbrush', 'kite', 'keycard', 
        'heron', 'canteen', 'soap', 'stirrer', 'shawl', 'shower_head', 'corkboard', 'identity_card', 'postcard', 'booklet', 
        'crossbar', 'fume_hood', 'pegboard', 'bulletin_board', 'coatrack', 'coloring_material', 'license_plate', 
        'paperback_book', 'triangle_(musical_instrument)', 'birthday_card', 'chopstick', 'paperweight', 'strap', 'dishtowel', 
        'passport', 'pennant', 'syringe', 'coat_hanger', 'frisbee', 'handkerchief', 'knocker_(on_a_door)', 'receipt', 
        'bath_mat', 'cleat_(for_securing_rope)', 'cover', 'dishrag', 'doorknob', 'vent', 'crayon', 'fishing_rod', 'bath_towel', 
        'rolling_pin', 'toothpick', 'rubber_band', 'towel', 'wind_chime', 'hand_towel', 'handle', 'tarp', 'bead', 'cincture', 
        'cornmeal', 'hinge', 'knitting_needle', 'parchment', 'bookmark', 'cigarette_case', 'funnel', 'matchbox', 'paper_plate', 
        'pencil_box'
        ]
    lvis_annotations = objaverse.load_lvis_annotations()
    class_list = list(lvis_annotations.keys())
    class_num_list = [len(lvis_annotations[c]) for c in class_list]
    total_num = np.array(class_num_list).sum()
    print("total num is :", total_num)
    remove_uids = []
    count = 0
    for i, c in enumerate(class_list):
        if c in filter_list:
            remove_uids.extend(lvis_annotations[c])
            count += 1
    print('remove class num: ', count, 'filter list num: ', len(filter_list))
            
    return remove_uids

if __name__ == '__main__':
    # prompt_file = './data/shapenet_chair/text_captions.csv'
    # prompt_list = []
    # with open(prompt_file, 'r') as f:
    #     csv_reader = csv.reader(f)
    #     for row in csv_reader:
    #         prompt_list.append(row)
    # prompt_dict = {} 
    # for p in prompt_list:
    #     if p[4] == '03001627' and p[1] not in prompt_dict:
    #         prompt_dict[p[1]] = p[2]

    path = './dataset/objaverse_dep_256'
    objs = os.listdir(path)
    num = len(objs)
    print('total num: ', num)
    train_objs = objs
    prompts = {}
    with open('./data/objaverse_40k/Cap3D_automated_Objaverse.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            prompts[row[0]] = row[1]
    print('Loaded all prompts.')
    count = 0
    with open('./data/objaverse_37k/train.txt', 'w') as f:
        for obj in train_objs:
            if os.path.exists(os.path.join(path, obj, '007.png')) and (obj in prompts): # 
                f.write(obj + '\n')
                count += 1
    print('total prompt image: ', count)

    # filtered_uids = objaverse_lvis_filter()
    # with open('./data/objaverse_40k/train.txt', 'r') as f:
    #     train_objs = [fn.strip() for fn in f.readlines()]
    # filtered_objs = []
    # count = 0
    # for obj in train_objs:
    #     if obj not in filtered_uids:
    #         filtered_objs.append(obj)
    #     else:
    #         count += 1
    # print('filtered num: ', len(filtered_objs), 'removed num:', count)
    # with open('./data/objaverse_40k/train_filtered.txt', 'w') as f:
    #     for obj in filtered_objs:
    #         f.write(obj + '\n')


