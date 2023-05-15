# Applications using ADIOS2

## Gray-Scott reaction diffusion model

The Gray-Scott system is a reaction-diffusion system. This means that it models a process that consists of a reaction and diffusion. In the case of the Gray-Scott model that reaction is a chemical reaction between two substances `U` and `V`, both of which diffuse over time. During the reaction `U` gets used up, while `V` is produced. 

The system is characterised by two parameters: 
- `F` the rate at which `U` is replenished
- `k` the rate at which `V` is removed from the system

Both substances diffuse over time at the diffusion rates `Du` and `Dv`.

The code with ADIOS is a 3D 7-point stencil code to simulate the Gray-Scott reaction diffusion model.

```
u_t = Du * (u_xx + u_yy + u_zz) - u * v^2 + F * (1 - u)  + noise * randn(-1,1)
v_t = Dv * (v_xx + v_yy + v_zz) + u * v^2 - (F + k) * v
```

Gray-Scott with SST video: https://users.nccs.gov/~pnorbert/GrayScottInsitu.mp4

ADIOS code in the examples repo: https://github.com/ornladios/ADIOS2-Examples/tree/master/source/cpp/gray-scott

### Gray-Scott coupled with an AI training code

Link: [https://github.com/coscholz1984/GS_CNN](https://github.com/coscholz1984/GS_CNN)

**More details on the Gray-Scott performance with ADIOS2 in the dedicated folder**
