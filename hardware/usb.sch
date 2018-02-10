EESchema Schematic File Version 4
LIBS:power
LIBS:Device
LIBS:Connector_Specialized
LIBS:Connector_Generic
LIBS:Logic_74xgxx
LIBS:Logic_CMOS_4000
LIBS:Valve
LIBS:fmcw3
LIBS:Switch
EELAYER 26 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 3 10
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Device:R R80
U 1 1 583B312F
P 4400 3100
F 0 "R80" V 4480 3100 50  0000 C CNN
F 1 "0" V 4400 3100 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 4330 3100 30  0001 C CNN
F 3 "" H 4400 3100 30  0000 C CNN
	1    4400 3100
	0    1    1    0   
$EndComp
$Comp
L Device:R R81
U 1 1 583B315C
P 4400 3250
F 0 "R81" V 4480 3250 50  0000 C CNN
F 1 "0" V 4400 3250 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 4330 3250 30  0001 C CNN
F 3 "" H 4400 3250 30  0000 C CNN
	1    4400 3250
	0    1    1    0   
$EndComp
Text Label 4100 3250 2    60   ~ 0
USBDP
Text Label 4100 3100 2    60   ~ 0
USBDM
$Comp
L Device:C C210
U 1 1 583B35D4
P 4550 2550
F 0 "C210" H 4575 2650 50  0000 L CNN
F 1 "100n" H 4575 2450 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 4588 2400 30  0001 C CNN
F 3 "" H 4550 2550 60  0000 C CNN
	1    4550 2550
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR040
U 1 1 583B35F7
P 4550 2800
F 0 "#PWR040" H 4550 2550 50  0001 C CNN
F 1 "GND" H 4550 2650 50  0000 C CNN
F 2 "" H 4550 2800 60  0000 C CNN
F 3 "" H 4550 2800 60  0000 C CNN
	1    4550 2800
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR041
U 1 1 583B4921
P 6300 6200
F 0 "#PWR041" H 6300 5950 50  0001 C CNN
F 1 "GND" H 6300 6050 50  0000 C CNN
F 2 "" H 6300 6200 60  0000 C CNN
F 3 "" H 6300 6200 60  0000 C CNN
	1    6300 6200
	1    0    0    -1  
$EndComp
$Comp
L Device:C C212
U 1 1 583B55E3
P 6200 1250
F 0 "C212" H 6225 1350 50  0000 L CNN
F 1 "100n" H 6225 1150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 6238 1100 30  0001 C CNN
F 3 "" H 6200 1250 60  0000 C CNN
	1    6200 1250
	1    0    0    -1  
$EndComp
$Comp
L Device:C C211
U 1 1 583B5665
P 5600 1600
F 0 "C211" H 5625 1700 50  0000 L CNN
F 1 "100n" H 5625 1500 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 5638 1450 30  0001 C CNN
F 3 "" H 5600 1600 60  0000 C CNN
	1    5600 1600
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR042
U 1 1 583B5768
P 6200 1400
F 0 "#PWR042" H 6200 1150 50  0001 C CNN
F 1 "GND" H 6200 1250 50  0000 C CNN
F 2 "" H 6200 1400 60  0000 C CNN
F 3 "" H 6200 1400 60  0000 C CNN
	1    6200 1400
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR043
U 1 1 583B5791
P 5600 1750
F 0 "#PWR043" H 5600 1500 50  0001 C CNN
F 1 "GND" H 5600 1600 50  0000 C CNN
F 2 "" H 5600 1750 60  0000 C CNN
F 3 "" H 5600 1750 60  0000 C CNN
	1    5600 1750
	1    0    0    -1  
$EndComp
$Comp
L Device:R R84
U 1 1 583B5C84
P 8000 4000
F 0 "R84" V 8080 4000 50  0000 C CNN
F 1 "33" V 8000 4000 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 7930 4000 30  0001 C CNN
F 3 "" H 8000 4000 30  0000 C CNN
	1    8000 4000
	0    1    1    0   
$EndComp
$Comp
L Device:R R88
U 1 1 583B5CC5
P 8400 4100
F 0 "R88" V 8480 4100 50  0000 C CNN
F 1 "33" V 8400 4100 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 8330 4100 30  0001 C CNN
F 3 "" H 8400 4100 30  0000 C CNN
	1    8400 4100
	0    1    1    0   
$EndComp
$Comp
L Device:R R85
U 1 1 583B63A8
P 8000 4300
F 0 "R85" V 8080 4300 50  0000 C CNN
F 1 "33" V 8000 4300 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 7930 4300 30  0001 C CNN
F 3 "" H 8000 4300 30  0000 C CNN
	1    8000 4300
	0    1    1    0   
$EndComp
Text HLabel 8800 4000 2    60   Output ~ 0
TCK
Text HLabel 8800 4100 2    60   Output ~ 0
TDI
Text HLabel 8800 4200 2    60   Input ~ 0
TDO
Text HLabel 8800 4300 2    60   Output ~ 0
TMS
Text HLabel 8150 2300 2    60   BiDi ~ 0
D0
Text HLabel 8150 2400 2    60   BiDi ~ 0
D1
Text HLabel 8150 2500 2    60   BiDi ~ 0
D2
Text HLabel 8150 2600 2    60   BiDi ~ 0
D3
Text HLabel 8150 2700 2    60   BiDi ~ 0
D4
Text HLabel 8150 2800 2    60   BiDi ~ 0
D5
Text HLabel 8150 2900 2    60   BiDi ~ 0
D6
Text HLabel 8150 3000 2    60   BiDi ~ 0
D7
Text HLabel 8150 3150 2    60   Output ~ 0
RXF#
Text HLabel 8150 3250 2    60   Output ~ 0
TXE#
Text HLabel 8150 3350 2    60   Input ~ 0
RD#
Text HLabel 8150 3450 2    60   Input ~ 0
WR
Text HLabel 8150 3550 2    60   Input ~ 0
SIWUA
$Comp
L fmcw3:USB-MICRO U31
U 1 1 583BA416
P 1300 1950
F 0 "U31" H 1250 2000 60  0000 C CNN
F 1 "USB-MICRO" H 1150 2250 60  0000 C CNN
F 2 "fmcw3:USB_MICRO" H 1350 1700 60  0001 C CNN
F 3 "" H 1350 1700 60  0000 C CNN
	1    1300 1950
	1    0    0    -1  
$EndComp
Text Label 2100 1700 0    60   ~ 0
USB_5V
Text Label 2100 1800 0    60   ~ 0
USBDM
Text Label 2100 1900 0    60   ~ 0
USBDP
$Comp
L power:GND #PWR044
U 1 1 583BA75D
P 2200 2250
F 0 "#PWR044" H 2200 2000 50  0001 C CNN
F 1 "GND" H 2200 2100 50  0000 C CNN
F 2 "" H 2200 2250 60  0000 C CNN
F 3 "" H 2200 2250 60  0000 C CNN
	1    2200 2250
	1    0    0    -1  
$EndComp
NoConn ~ 1950 2200
$Comp
L fmcw3:XTAL_SMD4 Y1
U 1 1 583C5AF8
P 3850 4950
F 0 "Y1" H 3850 5100 50  0000 C CNN
F 1 "ABM10-167-12.000MHZ-T3" V 3850 5700 50  0000 C CNN
F 2 "fmcw3:ABM10" H 3850 4950 60  0001 C CNN
F 3 "" H 3850 4950 60  0000 C CNN
	1    3850 4950
	0    1    1    0   
$EndComp
$Comp
L Device:C C209
U 1 1 583C5B9D
P 3700 5250
F 0 "C209" H 3725 5350 50  0000 L CNN
F 1 "15p" H 3725 5150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 3738 5100 30  0001 C CNN
F 3 "" H 3700 5250 60  0000 C CNN
	1    3700 5250
	0    1    1    0   
$EndComp
$Comp
L Device:C C208
U 1 1 583C5BF7
P 3700 4600
F 0 "C208" H 3725 4700 50  0000 L CNN
F 1 "15p" H 3725 4500 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 3738 4450 30  0001 C CNN
F 3 "" H 3700 4600 60  0000 C CNN
	1    3700 4600
	0    1    1    0   
$EndComp
$Comp
L power:GND #PWR045
U 1 1 583C6491
P 3350 4600
F 0 "#PWR045" H 3350 4350 50  0001 C CNN
F 1 "GND" H 3350 4450 50  0000 C CNN
F 2 "" H 3350 4600 60  0000 C CNN
F 3 "" H 3350 4600 60  0000 C CNN
	1    3350 4600
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR046
U 1 1 583C64DF
P 3350 5250
F 0 "#PWR046" H 3350 5000 50  0001 C CNN
F 1 "GND" H 3350 5100 50  0000 C CNN
F 2 "" H 3350 5250 60  0000 C CNN
F 3 "" H 3350 5250 60  0000 C CNN
	1    3350 5250
	1    0    0    -1  
$EndComp
$Comp
L fmcw3:93LC46B U32
U 1 1 583C7AB4
P 3900 6800
F 0 "U32" H 4050 6550 60  0000 C CNN
F 1 "93LC46B" H 3950 7050 60  0000 C CNN
F 2 "fmcw3:SOT-23-6" H 4000 6400 60  0001 C CNN
F 3 "" H 4000 6400 60  0001 C CNN
	1    3900 6800
	1    0    0    -1  
$EndComp
Text Label 4500 6800 0    60   ~ 0
EECS
Text Label 4500 6900 0    60   ~ 0
EECLK
$Comp
L Device:R R73
U 1 1 583C7CCB
P 1300 4200
F 0 "R73" V 1380 4200 50  0000 C CNN
F 1 "10k" V 1300 4200 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 1230 4200 30  0001 C CNN
F 3 "" H 1300 4200 30  0000 C CNN
	1    1300 4200
	1    0    0    -1  
$EndComp
$Comp
L Device:R R79
U 1 1 583C7D2C
P 1300 4600
F 0 "R79" V 1380 4600 50  0000 C CNN
F 1 "2.2k" V 1300 4600 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 1230 4600 30  0001 C CNN
F 3 "" H 1300 4600 30  0000 C CNN
	1    1300 4600
	1    0    0    -1  
$EndComp
Text Label 1400 4400 0    60   ~ 0
DO
Text Label 1350 4900 0    60   ~ 0
EEDATA
Text Label 3350 6700 2    60   ~ 0
DO
$Comp
L power:GND #PWR047
U 1 1 583C8ACF
P 2850 6900
F 0 "#PWR047" H 2850 6650 50  0001 C CNN
F 1 "GND" H 2850 6750 50  0000 C CNN
F 2 "" H 2850 6900 60  0000 C CNN
F 3 "" H 2850 6900 60  0000 C CNN
	1    2850 6900
	1    0    0    -1  
$EndComp
Text Label 4350 6150 0    60   ~ 0
3V3D
$Comp
L Device:C C207
U 1 1 583C8C2A
P 4600 6300
F 0 "C207" H 4625 6400 50  0000 L CNN
F 1 "100n" H 4625 6200 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 4638 6150 30  0001 C CNN
F 3 "" H 4600 6300 60  0000 C CNN
	1    4600 6300
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR048
U 1 1 583C8C93
P 4600 6450
F 0 "#PWR048" H 4600 6200 50  0001 C CNN
F 1 "GND" H 4600 6300 50  0000 C CNN
F 2 "" H 4600 6450 60  0000 C CNN
F 3 "" H 4600 6450 60  0000 C CNN
	1    4600 6450
	1    0    0    -1  
$EndComp
Text Label 1300 4000 0    60   ~ 0
3V3D
Text Label 3350 6900 2    60   ~ 0
EEDATA
$Comp
L Device:L_Core_Ferrite FB12
U 1 1 5865C0D7
P 4550 1300
AR Path="/5865C0D7" Ref="FB12"  Part="1" 
AR Path="/59395D6A/583ABBE9/5865C0D7" Ref="FB12"  Part="1" 
F 0 "FB12" V 4650 1300 50  0000 C CNN
F 1 "BLM18PG181SN1D" V 4500 1300 50  0000 C CNN
F 2 "fmcw3:C_0603b" H 4550 1300 60  0001 C CNN
F 3 "" H 4550 1300 60  0000 C CNN
	1    4550 1300
	0    -1   -1   0   
$EndComp
NoConn ~ 1950 2000
$Comp
L Device:L_Core_Ferrite FB14
U 1 1 596BB650
P 4550 1050
AR Path="/596BB650" Ref="FB14"  Part="1" 
AR Path="/59395D6A/583ABBE9/596BB650" Ref="FB14"  Part="1" 
F 0 "FB14" V 4650 1050 50  0000 C CNN
F 1 "BLM18PG181SN1D" V 4500 1050 50  0000 C CNN
F 2 "fmcw3:C_0603b" H 4550 1050 60  0001 C CNN
F 3 "" H 4550 1050 60  0000 C CNN
	1    4550 1050
	0    -1   -1   0   
$EndComp
$Comp
L Device:C C158
U 1 1 596BBF3A
P 7150 1250
F 0 "C158" H 7175 1350 50  0000 L CNN
F 1 "100n" H 7175 1150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 7188 1100 30  0001 C CNN
F 3 "" H 7150 1250 60  0000 C CNN
	1    7150 1250
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR049
U 1 1 596BBF40
P 7150 1400
F 0 "#PWR049" H 7150 1150 50  0001 C CNN
F 1 "GND" H 7150 1250 50  0000 C CNN
F 2 "" H 7150 1400 60  0000 C CNN
F 3 "" H 7150 1400 60  0000 C CNN
	1    7150 1400
	1    0    0    -1  
$EndComp
$Comp
L Device:C C159
U 1 1 596BC02C
P 7400 1250
F 0 "C159" H 7425 1350 50  0000 L CNN
F 1 "100n" H 7425 1150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 7438 1100 30  0001 C CNN
F 3 "" H 7400 1250 60  0000 C CNN
	1    7400 1250
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR050
U 1 1 596BC033
P 7400 1400
F 0 "#PWR050" H 7400 1150 50  0001 C CNN
F 1 "GND" H 7400 1250 50  0000 C CNN
F 2 "" H 7400 1400 60  0000 C CNN
F 3 "" H 7400 1400 60  0000 C CNN
	1    7400 1400
	1    0    0    -1  
$EndComp
$Comp
L Device:C C160
U 1 1 596BC066
P 7650 1250
F 0 "C160" H 7675 1350 50  0000 L CNN
F 1 "100n" H 7675 1150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 7688 1100 30  0001 C CNN
F 3 "" H 7650 1250 60  0000 C CNN
	1    7650 1250
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR051
U 1 1 596BC06D
P 7650 1400
F 0 "#PWR051" H 7650 1150 50  0001 C CNN
F 1 "GND" H 7650 1250 50  0000 C CNN
F 2 "" H 7650 1400 60  0000 C CNN
F 3 "" H 7650 1400 60  0000 C CNN
	1    7650 1400
	1    0    0    -1  
$EndComp
$Comp
L Device:C C161
U 1 1 596BC0A3
P 7900 1250
F 0 "C161" H 7925 1350 50  0000 L CNN
F 1 "100n" H 7925 1150 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 7938 1100 30  0001 C CNN
F 3 "" H 7900 1250 60  0000 C CNN
	1    7900 1250
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR052
U 1 1 596BC0AA
P 7900 1400
F 0 "#PWR052" H 7900 1150 50  0001 C CNN
F 1 "GND" H 7900 1250 50  0000 C CNN
F 2 "" H 7900 1400 60  0000 C CNN
F 3 "" H 7900 1400 60  0000 C CNN
	1    7900 1400
	1    0    0    -1  
$EndComp
Text HLabel 8050 1050 2    60   Input ~ 0
3V3D
$Comp
L Device:C C155
U 1 1 596BD0A1
P 4950 1650
F 0 "C155" H 4975 1750 50  0000 L CNN
F 1 "4.7u" H 4975 1550 50  0000 L CNN
F 2 "fmcw3:C_0603b" H 4988 1500 30  0001 C CNN
F 3 "" H 4950 1650 60  0000 C CNN
	1    4950 1650
	1    0    0    -1  
$EndComp
$Comp
L Device:C C157
U 1 1 596BD0F6
P 5200 1650
F 0 "C157" H 5225 1750 50  0000 L CNN
F 1 "4.7u" H 5225 1550 50  0000 L CNN
F 2 "fmcw3:C_0603b" H 5238 1500 30  0001 C CNN
F 3 "" H 5200 1650 60  0000 C CNN
	1    5200 1650
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR053
U 1 1 596BD1A8
P 4950 1800
F 0 "#PWR053" H 4950 1550 50  0001 C CNN
F 1 "GND" H 4950 1650 50  0000 C CNN
F 2 "" H 4950 1800 60  0000 C CNN
F 3 "" H 4950 1800 60  0000 C CNN
	1    4950 1800
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR054
U 1 1 596BD1F2
P 5200 1800
F 0 "#PWR054" H 5200 1550 50  0001 C CNN
F 1 "GND" H 5200 1650 50  0000 C CNN
F 2 "" H 5200 1800 60  0000 C CNN
F 3 "" H 5200 1800 60  0000 C CNN
	1    5200 1800
	1    0    0    -1  
$EndComp
Wire Wire Line
	4550 3100 5150 3100
Wire Wire Line
	4550 3250 5150 3250
Wire Wire Line
	4100 3100 4250 3100
Wire Wire Line
	4250 3250 4100 3250
Wire Wire Line
	4550 2400 5150 2400
Wire Wire Line
	6300 6050 6300 6150
Wire Wire Line
	5150 6150 5850 6150
Wire Wire Line
	6400 6150 6400 6050
Connection ~ 6300 6150
Wire Wire Line
	6500 6150 6500 6050
Connection ~ 6400 6150
Wire Wire Line
	6600 6150 6600 6050
Connection ~ 6500 6150
Wire Wire Line
	6700 6150 6700 6050
Connection ~ 6600 6150
Wire Wire Line
	7850 4000 7550 4000
Wire Wire Line
	7550 4100 8250 4100
Wire Wire Line
	7550 4200 8800 4200
Wire Wire Line
	8150 4000 8800 4000
Wire Wire Line
	8550 4100 8800 4100
Wire Wire Line
	7550 4300 7850 4300
Wire Wire Line
	8150 4300 8800 4300
Wire Wire Line
	7550 2300 8150 2300
Wire Wire Line
	7550 2400 8150 2400
Wire Wire Line
	7550 2500 8150 2500
Wire Wire Line
	7550 2600 8150 2600
Wire Wire Line
	7550 2700 8150 2700
Wire Wire Line
	7550 2800 8150 2800
Wire Wire Line
	7550 2900 8150 2900
Wire Wire Line
	7550 3000 8150 3000
Wire Wire Line
	7550 3150 8150 3150
Wire Wire Line
	7550 3250 8150 3250
Wire Wire Line
	7550 3350 8150 3350
Wire Wire Line
	7550 3450 8150 3450
Wire Wire Line
	7550 3550 8150 3550
Wire Wire Line
	2100 1700 1950 1700
Wire Wire Line
	2100 1800 1950 1800
Wire Wire Line
	2100 1900 1950 1900
Wire Wire Line
	1950 2100 2200 2100
Wire Wire Line
	2200 2100 2200 2250
Wire Wire Line
	3850 4600 3850 4750
Wire Wire Line
	3850 5100 3850 5150
Wire Wire Line
	3850 4750 5200 4750
Connection ~ 3850 4750
Wire Wire Line
	3850 5150 4850 5150
Connection ~ 3850 5150
Wire Wire Line
	3350 5250 3550 5250
Wire Wire Line
	3350 4600 3550 4600
Wire Wire Line
	1300 4350 1300 4400
Wire Wire Line
	4500 6800 4350 6800
Wire Wire Line
	4500 6900 4350 6900
Wire Wire Line
	3350 6900 3450 6900
Wire Wire Line
	3450 6700 3350 6700
Wire Wire Line
	3450 6800 2850 6800
Wire Wire Line
	2850 6800 2850 6900
Wire Wire Line
	4350 6150 4350 6700
Wire Wire Line
	4600 6150 4350 6150
Wire Wire Line
	1300 4050 1300 4000
Wire Wire Line
	1300 4750 1300 4900
Wire Wire Line
	1300 4900 1350 4900
Wire Wire Line
	1400 4400 1300 4400
Connection ~ 1300 4400
Wire Wire Line
	6200 6050 6200 6150
Wire Wire Line
	6100 6050 6100 6150
Connection ~ 6200 6150
Wire Wire Line
	6000 6050 6000 6150
Connection ~ 6100 6150
Wire Wire Line
	5850 6050 5850 6150
Connection ~ 6000 6150
Wire Wire Line
	4550 2800 4550 2700
Wire Wire Line
	5850 1300 5850 2050
Wire Wire Line
	5950 1050 5950 2050
Connection ~ 5950 1050
Wire Wire Line
	6200 1050 6200 1100
Wire Wire Line
	5600 1450 5600 1300
Connection ~ 5600 1300
Wire Wire Line
	7150 1050 7150 1100
Wire Wire Line
	6550 1050 7150 1050
Wire Wire Line
	7400 1050 7400 1100
Wire Wire Line
	7650 1050 7650 1100
Wire Wire Line
	7900 1050 7900 1100
Wire Wire Line
	6550 1050 6550 1900
Connection ~ 7150 1050
Wire Wire Line
	6450 2050 6450 1900
Wire Wire Line
	6450 1900 6550 1900
Connection ~ 6550 1900
Wire Wire Line
	6650 1900 6650 2050
Wire Wire Line
	6750 1900 6750 2050
Connection ~ 6650 1900
Connection ~ 7400 1050
Connection ~ 7650 1050
Connection ~ 7900 1050
Wire Wire Line
	4950 1500 4950 1300
Connection ~ 4950 1300
Wire Wire Line
	5200 1500 5200 1050
Connection ~ 5200 1050
Text Label 4600 2400 0    60   ~ 0
3V3D
Wire Wire Line
	5150 2500 4950 2500
Text Label 4950 2500 0    60   ~ 0
1V8
$Comp
L Device:C C156
U 1 1 596BE99F
P 4950 2650
F 0 "C156" H 4975 2750 50  0000 L CNN
F 1 "4.7u" H 4975 2550 50  0000 L CNN
F 2 "fmcw3:C_0603b" H 4988 2500 30  0001 C CNN
F 3 "" H 4950 2650 60  0000 C CNN
	1    4950 2650
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR055
U 1 1 596BEA28
P 4950 2800
F 0 "#PWR055" H 4950 2550 50  0001 C CNN
F 1 "GND" H 4950 2650 50  0000 C CNN
F 2 "" H 4950 2800 60  0000 C CNN
F 3 "" H 4950 2800 60  0000 C CNN
	1    4950 2800
	1    0    0    -1  
$EndComp
Text Label 4100 1050 2    60   ~ 0
3V3D
Text Label 4100 1300 2    60   ~ 0
3V3D
$Comp
L Device:R R72
U 1 1 596BF8E7
P 4850 3650
F 0 "R72" V 4930 3650 50  0000 C CNN
F 1 "1k" V 4850 3650 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 4780 3650 30  0001 C CNN
F 3 "" H 4850 3650 30  0000 C CNN
	1    4850 3650
	0    1    1    0   
$EndComp
Wire Wire Line
	5000 3650 5150 3650
Wire Wire Line
	4700 3650 4600 3650
Text Label 4600 3650 2    60   ~ 0
3V3D
Wire Wire Line
	5150 3500 4250 3500
Wire Wire Line
	4250 3500 4250 3600
$Comp
L Device:R R71
U 1 1 596C00A3
P 4250 3750
F 0 "R71" V 4330 3750 50  0000 C CNN
F 1 "12k" V 4250 3750 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 4180 3750 30  0001 C CNN
F 3 "" H 4250 3750 30  0000 C CNN
	1    4250 3750
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR056
U 1 1 596C016D
P 4250 3900
F 0 "#PWR056" H 4250 3650 50  0001 C CNN
F 1 "GND" H 4250 3750 50  0000 C CNN
F 2 "" H 4250 3900 60  0000 C CNN
F 3 "" H 4250 3900 60  0000 C CNN
	1    4250 3900
	1    0    0    -1  
$EndComp
Wire Wire Line
	5150 5650 5150 6150
Connection ~ 5850 6150
Wire Wire Line
	5150 4100 5050 4100
Wire Wire Line
	5150 4200 5050 4200
Wire Wire Line
	5150 4300 5050 4300
Text Label 5050 4100 2    60   ~ 0
EECS
Text Label 5050 4200 2    60   ~ 0
EECLK
Text Label 5050 4300 2    60   ~ 0
EEDATA
Wire Wire Line
	5150 5450 4850 5450
Wire Wire Line
	4850 5450 4850 5150
$Comp
L Device:R R82
U 1 1 596C316E
P 8600 5800
F 0 "R82" V 8680 5800 50  0000 C CNN
F 1 "4.7k" V 8600 5800 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 8530 5800 30  0001 C CNN
F 3 "" H 8600 5800 30  0000 C CNN
	1    8600 5800
	1    0    0    -1  
$EndComp
$Comp
L Device:R R83
U 1 1 596C31BB
P 8600 6150
F 0 "R83" V 8680 6150 50  0000 C CNN
F 1 "10k" V 8600 6150 50  0000 C CNN
F 2 "fmcw3:R_0402b" V 8530 6150 30  0001 C CNN
F 3 "" H 8600 6150 30  0000 C CNN
	1    8600 6150
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR057
U 1 1 596C327E
P 8600 6300
F 0 "#PWR057" H 8600 6050 50  0001 C CNN
F 1 "GND" H 8600 6150 50  0000 C CNN
F 2 "" H 8600 6300 60  0000 C CNN
F 3 "" H 8600 6300 60  0000 C CNN
	1    8600 6300
	1    0    0    -1  
$EndComp
Wire Wire Line
	8600 5950 8600 6000
Text Label 8650 5600 0    60   ~ 0
USB_5V
Wire Wire Line
	6300 1950 6300 2050
Wire Wire Line
	6100 1950 6200 1950
Wire Wire Line
	6200 1950 6200 2050
Wire Wire Line
	6100 1950 6100 2050
Connection ~ 6200 1950
Text Label 6100 1950 0    60   ~ 0
1V8
$Comp
L Device:C C152
U 1 1 596C4A23
P 3750 1900
F 0 "C152" H 3775 2000 50  0000 L CNN
F 1 "100n" H 3775 1800 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 3788 1750 30  0001 C CNN
F 3 "" H 3750 1900 60  0000 C CNN
	1    3750 1900
	1    0    0    -1  
$EndComp
$Comp
L Device:C C153
U 1 1 596C4AA8
P 3950 1900
F 0 "C153" H 3975 2000 50  0000 L CNN
F 1 "100n" H 3975 1800 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 3988 1750 30  0001 C CNN
F 3 "" H 3950 1900 60  0000 C CNN
	1    3950 1900
	1    0    0    -1  
$EndComp
$Comp
L Device:C C154
U 1 1 596C4AFE
P 4150 1900
F 0 "C154" H 4175 2000 50  0000 L CNN
F 1 "100n" H 4175 1800 50  0000 L CNN
F 2 "fmcw3:C_0402b" H 4188 1750 30  0001 C CNN
F 3 "" H 4150 1900 60  0000 C CNN
	1    4150 1900
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR058
U 1 1 596C4CA2
P 4150 2050
F 0 "#PWR058" H 4150 1800 50  0001 C CNN
F 1 "GND" H 4150 1900 50  0000 C CNN
F 2 "" H 4150 2050 60  0000 C CNN
F 3 "" H 4150 2050 60  0000 C CNN
	1    4150 2050
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR059
U 1 1 596C4CF6
P 3950 2050
F 0 "#PWR059" H 3950 1800 50  0001 C CNN
F 1 "GND" H 3950 1900 50  0000 C CNN
F 2 "" H 3950 2050 60  0000 C CNN
F 3 "" H 3950 2050 60  0000 C CNN
	1    3950 2050
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR060
U 1 1 596C4D43
P 3750 2050
F 0 "#PWR060" H 3750 1800 50  0001 C CNN
F 1 "GND" H 3750 1900 50  0000 C CNN
F 2 "" H 3750 2050 60  0000 C CNN
F 3 "" H 3750 2050 60  0000 C CNN
	1    3750 2050
	1    0    0    -1  
$EndComp
Wire Wire Line
	3750 1750 3750 1700
Wire Wire Line
	3750 1700 3950 1700
Wire Wire Line
	3950 1700 3950 1750
Wire Wire Line
	4150 1700 4150 1750
Connection ~ 3950 1700
Text Label 3800 1700 0    60   ~ 0
1V8
Wire Wire Line
	7550 5550 8400 5550
Wire Wire Line
	8400 5550 8400 5950
Wire Wire Line
	8400 5950 8600 5950
$Comp
L fmcw3:FT2232H U33
U 1 1 596B9CA6
P 6250 4000
F 0 "U33" H 6250 3900 60  0000 C CNN
F 1 "FT2232H" H 6250 4100 60  0000 C CNN
F 2 "Package_QFP:LQFP-64_10x10mm_P0.5mm" H 6050 4300 60  0001 C CNN
F 3 "" H 6050 4300 60  0001 C CNN
	1    6250 4000
	1    0    0    -1  
$EndComp
Wire Wire Line
	7550 3650 8150 3650
Text HLabel 8150 3650 2    60   Output ~ 0
CLKOUT
Wire Wire Line
	7550 3750 8150 3750
Text HLabel 8150 3750 2    60   Input ~ 0
OE#
NoConn ~ 7550 3850
NoConn ~ 7550 5700
Wire Wire Line
	8650 5600 8600 5600
Wire Wire Line
	8600 5600 8600 5650
NoConn ~ 3700 4900
NoConn ~ 3700 5000
NoConn ~ 7550 4400
NoConn ~ 7550 4500
NoConn ~ 7550 4600
NoConn ~ 7550 4700
NoConn ~ 7550 4850
NoConn ~ 7550 4950
NoConn ~ 7550 5150
NoConn ~ 7550 5250
NoConn ~ 7550 5350
NoConn ~ 7550 5450
Wire Wire Line
	7550 5800 7600 5800
Text HLabel 7600 5800 2    60   Output ~ 0
SUSPEND
NoConn ~ 7550 5050
Wire Wire Line
	6300 6150 6300 6200
Wire Wire Line
	6300 6150 6400 6150
Wire Wire Line
	6400 6150 6500 6150
Wire Wire Line
	6500 6150 6600 6150
Wire Wire Line
	6600 6150 6700 6150
Wire Wire Line
	3850 4750 3850 4800
Wire Wire Line
	3850 5150 3850 5250
Wire Wire Line
	1300 4400 1300 4450
Wire Wire Line
	6200 6150 6300 6150
Wire Wire Line
	6100 6150 6200 6150
Wire Wire Line
	6000 6150 6100 6150
Wire Wire Line
	5950 1050 6200 1050
Wire Wire Line
	5600 1300 5850 1300
Wire Wire Line
	7150 1050 7400 1050
Wire Wire Line
	6550 1900 6550 2050
Wire Wire Line
	6550 1900 6650 1900
Wire Wire Line
	6650 1900 6750 1900
Wire Wire Line
	7400 1050 7650 1050
Wire Wire Line
	7650 1050 7900 1050
Wire Wire Line
	7900 1050 8050 1050
Wire Wire Line
	4950 1300 5600 1300
Wire Wire Line
	5200 1050 5950 1050
Wire Wire Line
	5850 6150 6000 6150
Wire Wire Line
	6200 1950 6300 1950
Wire Wire Line
	3950 1700 4150 1700
Wire Wire Line
	4700 1050 5200 1050
Wire Wire Line
	4100 1050 4400 1050
Wire Wire Line
	4100 1300 4400 1300
Wire Wire Line
	4700 1300 4950 1300
$EndSCHEMATC
