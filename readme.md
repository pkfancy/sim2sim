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
foot_names = ["FL", "FR", "RL", "RR"]
foot_pos_list = [d.geom_xpos[mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, s)] for s in foot_names]

# get joint positions
# d.qpos[7:19]

# groudtruth base position and rotation
# d.qpos[:3]
# d.qpos[3:7]
```

### Position Estimation Given the Rotation
将机器人基座在初始状态（位置和姿态和世界坐标系相同）按照给定的关节角度进行调节，得到四足位置 $\{x_2'\}$；然后将机器人基座按照给定姿态进行调整，则此时机器人的姿态和问题中的姿态相同；最后求四足在世界坐标系中的实际位置 $\{x_2\}$ 和上述位置之差 $\{x_2 - x_2'\}$，就是机器人基座的位置



### Position and Rotation Estimation
将机器人基座在初始状态（位置和姿态和世界坐标系相同）按照给定的关节角度进行调节，得到四足位置 $\{x_2'\}$；将此时的四足位置和给定的四足位置 $\{x_2\}$ 进行比较，相差一个刚体变换，对应的平移和旋转就是机器人基座的平移和旋转

具体的求解方式是对 $\{x_2\}, \{x_2'\}$ 进行中心化，则消去了基座平移，只差一个旋转；通过对四足坐标分别进行 SVD 可求出旋转矩阵


## Velocity Estimation
已知基座旋转角速度和各关节旋转角速度，求基座线速度


由于站立的脚满足 
$$v_{\text{foot}} = v_{\text{base}} + \omega_{\text{base}}\times r_{\text{foot in base}} + J \theta = 0$$
所以
$$v_{\text{base}} = -(\omega_{\text{base}}\times r_{\text{foot in base}} + J \theta)$$


```python
# base angular velocity
d.qvel[3:6]
# joint velocity
d.qvel[6:(6 + 12)]

# get stance feet
foot_pos[:, 2] < ground_height_th

# groudtruth base linear velocity
d.qvel[:3]
```