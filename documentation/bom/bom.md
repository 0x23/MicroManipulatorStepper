# Bill of Materials (BOM)

Printable parts link directly to their `.stl` file and PCBs to their production `.zip` (Gerber + drill files, ready to upload to a PCB manufacturer).

Both the PCBs and the 3D-printed parts can be ordered from [JLCPCB](https://jlcpcb.com), e.g. if you don't have a 3D printer.

## Mechanics

| Name | Qty | Description | Comments | Suggested link | Image |
|------|-----|-------------|----------|----------------|-------|
| Steel ball | 12 | Ø1 mm (0.04") | Precision G10 or better. Can be sourced from small ball bearings | [Aliexpress](https://www.aliexpress.com/item/1005008399505167.html) | <img src="images/SteelBall.jpg" alt="Steel ball" width="180"> |
| PU stretch cord | 1 m | Ø1mm | e.g. "Stretch Magic" or "Magic String" | [Aliexpress](https://www.aliexpress.com/item/1005006140225266.html) | <img src="images/PU_StretchCord.png" alt="Pu strech cord" width="180"> |
| Neodym Cylinder Magnet | 70 | Ø3mm h=5 mm | N52 Grade | [NeoMagnets](https://www.buyneomagnets.com/p/3mm-dia-x-5mm-thick-n52-neodymium-rare-earth-disc-magnets-100-pack/) | <img src="images/NeodymMagnet.jpg" alt="Neodym Cylinder Magnet" width="180"> |
| Brass Rod | 1 m | OD=Ø2mm | | [Aliexpress](https://www.aliexpress.com/item/1005006371614363.html) | <img src="images/BrassRod.jpg" alt="Brass Rod" width="180"> |
| M2-8 dowel | 9 | | | [Aliexpress](https://www.aliexpress.com/item/1005009401701158.html) | <img src="images/M2x8_DowelPin.jpg" alt="M2-8 dowel" width="180"> |
| SHCS M2-8 | 9 | | | | |
| Set Screw M3-6 | 3 | | | | |
| FHCS M3-6 | 3 | | | | |
| FHCS M3-10 | 9 | | | | |
| FHCS M3-20 | 12 | | | | |
| Crimp Beads | 12 | | Used to affix the ends of the PU cord | | |

## Electronics

| Name | Qty | Description | Comments | Suggested link | Image |
|------|-----|-------------|----------|----------------|-------|
| DC-DC Step Down Module | 1 | 5V Fixed Output Step-Down Buck Converter | | [Aliexpress](https://www.aliexpress.com/item/1005008257960729.html) | <img src="images/Fixed_DC_DC-StepDown.png" alt="DC-DC Step Down" width="180"> |
| (1N5819) OR (PMEG4010) | 1 | Diode (through-hole or SMD) | | | |
| Raspberry Pi Pico 2 | 1 | Microcontroller board | | [Aliexpress](https://www.aliexpress.com/item/1005007660023339.html) | <img src="images/RaspberryPiPico2.png" alt="Raspberry Pi Pico 2" width="180"> |
| MT6835 Magnetic Encoder Module | 3 | Magnetic encoder | | [Aliexpress](https://www.aliexpress.com/item/1005012795954973.html) | <img src="images/MT6835.png" alt="MT6835 Magnetic Encoder Module" width="180"> |
| TB6612FNG Driver | 3 | | | [Aliexpress](https://www.aliexpress.com/item/1005006246388728.html) | <img src="images/TB6612FNG.png" alt="TB6612FNG Driver" width="180"> |
| 470 µF Electrolytic Capacitor | 3 | > 100 µF | anything > 100µF will work well | | |
| 10 µF Ceramic Capacitor 0603 | 3 | Optional | | | |
| 4.7 k Resistor 0603 | 2 | Optional | | | |
| NEMA 17 Stepper Motor | 3 | Body ≤ 38 mm | Body length <= 38mm | [Aliexpress](https://www.aliexpress.com/item/4001359634598.html) | <img src="images/NEMA17_Stepper.jpg" alt="NEMA 17 Stepper Motor" width="180"> |
| JST XH2.54 4P connector | 3 | 4P Housing & Header 2.54mm pitch | J1, J2, J3 (motors) | [Aliexpress](https://www.aliexpress.com/item/1005007080333485.html) | <img src="images/JST_XH2.54_Connector.png" alt="JST XH2.54 Connector" width="180"> |
| XT60H Connector | 1 | 2P Male & Female pair 7.5mm pitch | Power connector | [Aliexpress](https://www.aliexpress.com/item/1005007605928815.html) | <img src="images/XT60H_Pairs.png" alt="XT60H Connector" width="180"> |
| PH2.0 4P Housing <br> PH2.0 8P Housing | 1 <br> 1 | 4P 2mm pitch Housing <br> 8P 2mm pitch Housing | J5 (tool/I2C)<br> J4 (encoders) | [Aliexpress](https://www.aliexpress.com/item/1005010491403413.html) | <img src="images/PH2_Housings.png" alt="PH2.0 Housings" width="180"> |
| PH2.0 4P Header <br> PH2.0 8P Header | 1 <br> 1 | 4P 2mm pitch Header <br> 8P 2mm pitch Header | J5 (tool/I2C) <br> J4 (encoders) | [Aliexpress](https://www.aliexpress.com/item/1005010491403413.html) | <img src="images/PH2_Headers.png" alt="PH2.0 Headers" width="180"> |

## PCBs

| Name | Qty | Description | Comments | File | Image |
| ---- | --- | ----------- | -------- | ---- | ----- |
| Control Board | 1 | | | [.zip](../../construction/v4_PCB_zip/PrecisionStepperController3X_v1.3.zip) | <img src="images/PCB_ControlBoard.png" alt="Control Board" width="180"> |
| Ball Joint Plate (balljoint_plate_pcb) | 6 | | | [.zip](../../construction/v4_PCB_zip/balljoint_plate_pcb_jlcpcb.zip) | <img src="images/PCB_BallJointPlate.png" alt="Ball Joint Plate" width="180"> |
| Encoder Board | 1 | | | [.zip](../../construction/v4_PCB_zip/EncoderBoard.zip) | <img src="images/PCB_EncoderBoard.png" alt="Encoder Board" width="180"> |

## 3D Printing

| Name | Qty | Description | Comments | File | Image |
| ---- | --- | ----------- | -------- | ---- | ----- |
| Motor Horn | 3 | | 0.4 mm+ nozzle may require post-processing | [.stl](../../construction/v4_3dprinting_stl/machine/MotorHorn.stl) | <img src="images/MotorHorn.png" alt="Motor Horn" width="180"> |
| RubberBandCollet | 12 | | 0.4 mm nozzle, 0.1mm Layer height + may require post-processing | [.stl](../../construction/v4_3dprinting_stl/machine/RubberBandCollet.stl) | <img src="images/RubberBandCollet.png" alt="Rubber Band Collet" width="180"> |
| Motor Mount | 3 | | | [.stl](../../construction/v4_3dprinting_stl/machine/MotorMount.stl) | <img src="images/MotorMount.png" alt="Motor Mount" width="180"> |
| End Effector | 1 | | | [.stl](../../construction/v4_3dprinting_stl/machine/EndEffector.stl) | <img src="images/EndEffector.png" alt="End Effector" width="180"> |
| Base Block | 1 | | | [.stl](../../construction/v4_3dprinting_stl/machine/BaseBlock.stl) | <img src="images/BaseBlock.png" alt="Base Block" width="180"> |

# Tool - Rod Sharpener

## 3D Printing

| Name | Qty | Description | Comments | File | Image |
| ---- | --- | ----------- | -------- | ---- | ----- |
| Rod Guide | 1 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_sharpener/RodGuide.stl) | <img src="images/Sharpener_RodGuide.png" alt="Rod Guide (Sharpener)" width="180"> |
| Blade Carrier | 1 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_sharpener/BladeCarrier.stl) | <img src="images/Sharpener_BladeCarrier.png" alt="Blade Carrier" width="180"> |

# Tool - Rod Length Setting Press

## PCBs

| Name | Qty | Description | Comments | File | Image |
| ---- | --- | ----------- | -------- | ---- | ----- |
| ball_alignment_disc_pcb | 1 | | | [.zip](../../construction/v4_PCB_zip/ball_alignment_disc_pcb.zip) | <img src="images/PCB_BallAlignmentDisc.png" alt="Ball Alignment Disc" width="180"> |

## 3D Printing

| Name | Qty | Description | Comments | File | Image |
| ---- | --- | ----------- | -------- | ---- | ----- |
| RodGrindingJigA | 1 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_length_setting_press/RodGrindingJigA.stl) | <img src="images/Press_RodGrindingJigA.png" alt="Rod Grinding Jig A" width="180"> |
| RodGrindingJigB | 1 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_length_setting_press/RodGrindingJigB.stl) | <img src="images/Press_RodGrindingJigB.png" alt="Rod Grinding Jig B" width="180"> |
| RodGuide | 1 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_length_setting_press/RodGuide.stl) | <img src="images/Press_RodGuide.png" alt="Rod Guide (Length Press)" width="180"> |
| RodCollet | 2 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_length_setting_press/RodCollet.stl) | <img src="images/Press_RodCollet.png" alt="Rod Collet" width="180"> |
| CaliperClamp | 2 | | | [.stl](../../construction/v4_3dprinting_stl/tools/rod_length_setting_press/CaliperClamp.stl) | <img src="images/Press_CaliperClamp.png" alt="Caliper Clamp" width="180"> |
