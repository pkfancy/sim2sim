# Robot Parameters Estimation

## Position and Rotation Estimation

in world coordintate system
|| position | rotation |
|-|-|-|
|robot base|$x$|$R$
|joint|$x_1$|-
|foot|$x_2$|-

<!-- - robot base: 
  - position: $x$
  - rotation: $R$ -->

```python
# get foot positions
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_pos_list = [d.site_xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, s)] for s in foot_names]

# get joint positions
# d.qpos[7:19]

# groudtruth base position and rotation
# d.qpos[:3]
# d.qpos[3:7]
```

### Position Estimation Given the Rotation
将机器人基座在初始状态（位置和姿态和世界坐标系相同）按照给定的关节角度进行调节，得到四足位置 $\{x_2'\}$；然后将机器人基座按照给定姿态进行调整，则此时机器人的姿态和问题中的姿态相同；最后求四足在世界坐标系中的实际位置 $\{x_2\}$ 和上述位置之差 $\{x_2 - x_2'\}$，就是机器人基座的位置



### Position and Rotation Estimation
将机器人基座在初始状态（位置和姿态和世界坐标系相同）按照给定的关节角度进行调节，得到四足位置 $\{x_2'\}$；将此时的四足位置和给定的四组位置进行比较，相差一个刚体变换，对应的平移和旋转就是机器人基座的平移和旋转


## Velocity Estimation

