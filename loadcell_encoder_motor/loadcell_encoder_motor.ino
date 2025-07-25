#include "HX711.h"

////////////////////////////////////////////
//// 로드셀 설정
#define DOUT1  6  // 로드셀 1 데이터핀 
#define CLK1  7   // 로드셀 1 클럭핀 

#define DOUT2  16  // 로드셀 2 데이터핀
#define CLK2  17   // 로드셀 2 클럭핀

#define DOUT3  14  // 로드셀 3 데이터핀
#define CLK3  15   // 로드셀 3 클럭핀

#define DOUT4  4  // 로드셀 4 데이터핀
#define CLK4  5   // 로드셀 4 클럭핀

HX711 scale1;
HX711 scale2;
HX711 scale3;
HX711 scale4;
////////////////////////////////////////////

////////////////////////////////////////////
//// 인코더 모터 설정
// 핀 설정
const int IN1 = 10;
const int IN2 = 9;
const int ENA = 11;
const int ENC_A = 2;
const int ENC_B = 3;

// PI 제어 변수
float targetRPM = 1200.0;
float currentRPM = 0.0;
float Kp = 0.05;
float Ki = 0.05;
float Kd = 0.01;
float integral = 0;
float previousRPM = 0;


// 모터제어 변수
int pwmOutput = 80;
volatile long encoderCount = 0;
unsigned long lastTime = 0;
long lastEncoder = 0;

// 모터 CPR
const float CPR = 380.0;

bool print_motor = true;
////////////////////////////////////////////

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
    setup_scale_2();
    setup_scale_3();
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
    // [LoadCell_{cell_number}] init
    if(input.startsWith("[LoadCell_")) {
      int cell_number = input.substring(10, 11).toInt();
      String command_part = input.substring(13); // "cal:value" or "tare"

      if (command_part.startsWith("cal:")) {
        Serial.print("set calibration factor: ");
        Serial.println(cell_number);

        // "cal:" 다음의 값을 float으로 추출
        String cal_value_str = command_part.substring(4);
        float calibration_factor = cal_value_str.toFloat();
        
        if(cell_number == 1) {
          set_calibration_scale(&scale1, calibration_factor);
        } else if(cell_number == 2) {
          set_calibration_scale(&scale2, calibration_factor);
        } else if(cell_number == 3) {
          set_calibration_scale(&scale3, calibration_factor);
        } else if(cell_number == 4) {
          set_calibration_scale(&scale4, calibration_factor);
        } else {
          Serial.println("Invalid cell number");
        }
      } else if (command_part.startsWith("tare")) {
        Serial.print("tare cell: ");
        Serial.println(cell_number);

        if(cell_number == 1) {
          set_tare_scale(&scale1);
        } else if(cell_number == 2) {
          set_tare_scale(&scale2);
        } else if(cell_number == 3) {
          set_tare_scale(&scale3);
        } else if(cell_number == 4) {
          set_tare_scale(&scale4);
        } else {
          Serial.println("Invalid cell number");
        }
      } else if (command_part.startsWith("init")) {
        Serial.print("initialize cell: ");
        Serial.println(cell_number);

        if(cell_number == 1) {
          initialize_scale(&scale1);
        } else if(cell_number == 2) {
          initialize_scale(&scale2);
        } else if(cell_number == 3) {
          initialize_scale(&scale3);
        } else if(cell_number == 4) {
          initialize_scale(&scale4);
        } else {
          Serial.println("Invalid cell number");
        }
      }
    }
  }
}

void print_cell_weight() {
  // [LoadCell_input] Cell 1: -57 | Cell 4: 100 ...
  Serial.print("[LoadCell_input] Cell 1: ");
  Serial.print(scale1.get_units(1), 3);
  Serial.print(" | Cell 2: ");
  Serial.print(scale2.get_units(1), 3);
  Serial.print(" | Cell 3: ");
  Serial.print(scale3.get_units(1), 3);
  Serial.print(" | Cell 4: ");
  Serial.println(scale4.get_units(1), 3);
}

void set_calibration_scale(HX711 *scale, float calibration_factor)
{
  scale->set_scale(calibration_factor); //Adjust to this calibration factor
}

void set_tare_scale(HX711 *scale)
{
  scale->tare(); //Reset the scale to 0
}

void initialize_scale(HX711 *scale)
{
  set_calibration_scale(scale, 1.0);
  set_tare_scale(scale);
}


// 인코더 모터 셋업
void setup_motor() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);

  // 인터럽트 함수 적용
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, RISING);
  // attachInterrupt(digitalPinToInterrupt(ENC_B), encoderISRB, RISING);

  // 모터 정방향
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  lastTime = millis();
}

// 인코더 모터 등각속도 제어


void control_motor() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;  // 초 단위
  if (dt >= 0.2) {  // 200ms 주기

    noInterrupts();
    long count = encoderCount;
    interrupts();

    long deltaCount = count - lastEncoder;
    lastEncoder = count;

    currentRPM = (deltaCount / CPR) / dt * 60.0;

    float error = targetRPM - currentRPM;
    integral += error * dt;
    float control = Kp * error + Ki * integral;

    pwmOutput = (int)control;
    pwmOutput = constrain(pwmOutput, 0, 255);

    analogWrite(ENA, pwmOutput);

    if ( print_motor ) {
      Serial.print("[Motor] ");
      Serial.print("RPM: ");
      Serial.print(currentRPM);
      Serial.print(" | PWM: ");
      Serial.println(pwmOutput);
    }
    

    lastTime = now;
  }
}

// 모터 isr 증가
void encoderISR() {
  // Serial.println("encoder ISR interrupted!!");
  encoderCount++;
}

void encoderISRB() {
    // Serial.println("encoder ISR-B interrupted!!");
}

void setup() {
  setup_serial();
  setup_scale();
  setup_motor();
  Serial.println("Initialize complete");
}

void loop() {
  check_calibration_scale();
  print_cell_weight();
  control_motor();
  delay(100);
}