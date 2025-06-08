#include "HX711.h"

#define DOUT1  6 //데이터핀 3번핀
#define CLK1  7   // 클럭핀 2번핀

#define DOUT2  9 //데이터핀 7번핀
#define CLK2  8   // 클럭핀 2번핀

#define DOUT3  10 //데이터핀 9번핀
#define CLK3  11   // 클럭핀 8번핀

#define DOUT4  4 //데이터핀 10번핀
#define CLK4  5   // 클럭핀 11번핀

HX711 scale1;
HX711 scale2;
HX711 scale3;
HX711 scale4;

void setup_serial()
{
  Serial.begin(9600);
  Serial.println("HX711 calibration sketch");
  Serial.println("Remove all weight from scale1");
  Serial.println("After readings begin, place known weight on scale1");
}

void setup_scale()
{
    setup_scale_1();
    // setup_scale_2();
    // setup_scale_3();
    setup_scale_4();
}

void setup_scale_1()
{
  scale1.begin(DOUT1, CLK1);
  scale1.set_scale();
  scale1.tare(); //Reset the scale1 to 0

  long zero_factor = scale1.read_average(); //Get a baseline reading
  Serial.print("cell 1 initialize! Zero factor: "); //This can be used to remove the need to tare the scale1. Useful in permanent scale1 projects.
  Serial.println(zero_factor);
}

void setup_scale_2()
{
  scale2.begin(DOUT2, CLK2);
  scale2.set_scale();
  scale2.tare(); //Reset the scale2 to 0

  long zero_factor = scale2.read_average(); //Get a baseline reading
  Serial.print("cell 2 initialize! Zero factor: "); //This can be used to remove the need to tare the scale2. Useful in permanent scale2 projects.
  Serial.println(zero_factor);
}

void setup_scale_3()
{
  scale3.begin(DOUT3, CLK3);
  scale3.set_scale();
  scale3.tare(); //Reset the scale3 to 0

  long zero_factor = scale3.read_average(); //Get a baseline reading
  Serial.print("cell 3 initialize! Zero factor: "); //This can be used to remove the need to tare the scale3. Useful in permanent scale3 projects.
  Serial.println(zero_factor);
}

void setup_scale_4()
{
  scale4.begin(DOUT4, CLK4);
  scale4.set_scale();
  scale4.tare(); //Reset the scale4 to 0

  long zero_factor = scale4.read_average(); //Get a baseline reading
  Serial.print("cell 4 initialize! Zero factor: "); //This can be used to remove the need to tare the scale4. Useful in permanent scale4 projects.
  Serial.println(zero_factor);
}

void check_calibration_scale() {
  if(Serial.available()) {
    
    String input = Serial.readStringUntil('\n');


    // Calibration Stream example: 
    // [LoadCell_{cell_number}] cal:{calibration_factor}
    // [LoadCell_{cell_number}] tare
    
    if(input.startsWith("[LoadCell_")) {
      int cell_number = input.substring(9, 10).toInt();
      int calibration_factor = input.substring(11).toInt();
      String command = input.substring(11);

      
      if(cell_number == 1) {
        if (command.startsWith("cal:")) {
          set_calibration_scale(scale1, calibration_factor);
        } else if(command.startsWith("tare")) {
          set_tare_scale(scale1);
        }
      } else if(cell_number == 2) {
        if (command.startsWith("cal:")) {
          set_calibration_scale(scale2, calibration_factor);
        } else if(command.startsWith("tare")) {
          set_tare_scale(scale2);
        }
      } else if(cell_number == 3) {
        if (command.startsWith("cal:")) {
          set_calibration_scale(scale3, calibration_factor);
        } else if(command.startsWith("tare")) {
          set_tare_scale(scale3);
        }
      } else if(cell_number == 4) {
        if (command.startsWith("cal:")) {
          set_calibration_scale(scale4, calibration_factor);
        } else if(command.startsWith("tare")) {
          set_tare_scale(scale4);
        }
      } else {
        Serial.println("Invalid cell number");
      }
    }
  }
}

void print_cell_weight() {
  // [LoadCell_input] Cell 1: -57 | Cell 4: 100 ...
  Serial.print("[LoadCell_input] Cell 1: ");
  Serial.print(scale1.get_units(), 1);
  // Serial.print(" | Cell 2: ");
  // Serial.print(scale2.get_units(), 1);
  // Serial.print(" | Cell 3: ");
  // Serial.print(scale3.get_units(), 1);
  Serial.print(" | Cell 4: ");
  Serial.println(scale4.get_units(), 1);
}

void set_calibration_scale(HX711 scale, int calibration_factor)
{
  scale.set_scale(calibration_factor); //Adjust to this calibration factor
}

void set_tare_scale(HX711 scale)
{
  scale.tare(); //Reset the scale to 0
}


void setup() {
  setup_serial();
  setup_scale();

  Serial.println("Initialize complete");
}

void loop() {
  check_calibration_scale();
  print_cell_weight();
  delay(100);
}