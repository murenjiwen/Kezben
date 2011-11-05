import bpy, bpy.ops
import random
from mathutils import Euler

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_pattern(pattern="Armature", extend=False)
bpy.ops.object.mode_set(mode='POSE')

bones = bpy.data.objects['Armature'].pose.bones

hand_parents = [bones['palm.L']]
hand_children = [bones['hand.L']]
while len(hand_parents) > 0:
    parent = hand_parents.pop()
    hand_parents.extend(parent.children)
    hand_children.extend(parent.children)

for bone in hand_children:
    range[bone]={[(0,0),(0,0),(0,0)]}
range[bones["hand.L"]][0]=(-72.0,77.0)
[bones["finger_middle_knuckle.L"],
bones["finger_index_knuckle.L"], 
bones["finger_index_01.L"],
bones["finger_middle_01.L"] ,
bones["finger_pinky_knuckle.L"], 
bones["finger_pinky_01.L"],
bones["finger_thumb_knuckle.L"], 
bones["finger_thumb_01.L"],
bones["finger_ring_knuckle.L"], 
bones["finger_ring_01.L"],
bones["finger_ring_02.L"] ,
bones["finger_ring_03.L"] ,
bones["finger_thumb_02.L"] ,
bones["finger_thumb_03.L"] ,
bones["finger_pinky_02.L"] ,
bones["finger_pinky_03.L"] ,
bones["finger_middle_02.L"] ,
bones["finger_middle_03.L"], 
bones["finger_index_02.L"], 
bones["finger_index_03.L"]]

for bone in hand_children:
   bone.rotation_mode = 'XYZ' 
   lock=bone.lock_rotation
   bone.rotation_euler.zero()
   xrange = range[bone]['x']
   yrange = range[bone]['y']
   zrange = range[bone]['z']
   x = random.random()*(xrange[1]-xrange[0])+xrange[0]
   y=0
   z=0
   x=x*pi/180
   y=y*pi/180
   z=z*pi/180
   bone.rotation_euler.rotate(Euler([x,y,z], 'XYZ'))