/*
 * Elegoo Smart Robot Car V4.0 - Serial Drive + Non-Blocking In-Place Turns
 * Single-knob empirical calibration. No delay() / no blocking.
 *
 * Serial Monitor: 9600 baud, line ending = "Newline".
 *   F | B | S
 *   left,90   right,45    (degrees, spin in place)
 *   S aborts a turn in progress.
 */

#define PIN_PWMA 5
#define PIN_PWMB 6
#define PIN_AIN1 7   // left direction  (HIGH = forward)
#define PIN_BIN1 8   // right direction (HIGH = forward)
#define PIN_STBY 3

const int DRIVE_SPEED = 150;
const int TURN_SPEED  = 255;   // keep high: too slow and the wheels stall instead of scrubbing

// ===========================================================
//  THE ONLY NUMBER YOU TUNE
//  Milliseconds to spin EXACTLY 90 degrees in place, on your floor, your battery.
//  HOW TO CALIBRATE (takes ~4 tries):
//    1. Put the car on the floor it'll actually run on. Decent battery.
//    2. Send  right,90  and watch how far it turns.
//    3. Turned too little? RAISE this number. Too far? LOWER it.
//    4. Repeat until right,90 is a clean quarter turn. Done.
//  Every other angle scales from this automatically.
// ===========================================================
unsigned long MS_FOR_90 = 600;

String inputLine = "";
bool turning = false;
unsigned long turnEndTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(PIN_PWMA, OUTPUT); pinMode(PIN_PWMB, OUTPUT);
  pinMode(PIN_AIN1, OUTPUT); pinMode(PIN_BIN1, OUTPUT);
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, HIGH);
  stopMotors();
  Serial.println("Ready. F, B, S, left,90, right,45");
}

void loop() {
  while (Serial.available() > 0) {                 // read serial every loop, never blocks
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputLine.length() > 0) { handleCommand(inputLine); inputLine = ""; }
    } else {
      inputLine += c;
    }
  }
  if (turning && millis() >= turnEndTime) {         // end the turn on a timer, no delay()
    stopMotors();
    turning = false;
    Serial.println("Turn complete");
  }
}

void handleCommand(String cmd) {
  cmd.trim(); cmd.toLowerCase();
  int comma = cmd.indexOf(',');

  if (comma == -1) {
    if (cmd == "f" || cmd == "forward")        { turning = false; forward();    Serial.println("Forward"); }
    else if (cmd == "b" || cmd == "backward")  { turning = false; backward();   Serial.println("Backward"); }
    else if (cmd == "s" || cmd == "stop")      { turning = false; stopMotors(); Serial.println("Stop"); }
    else Serial.println("Unknown: " + cmd);
    return;
  }

  String name = cmd.substring(0, comma);
  int degrees = cmd.substring(comma + 1).toInt();
  if (degrees <= 0) { Serial.println("Need positive degrees"); return; }

  if (name == "left")       startTurn(true,  degrees);
  else if (name == "right") startTurn(false, degrees);
  else Serial.println("Unknown: " + name);
}

void startTurn(bool left, int degrees) {
  if (left) { digitalWrite(PIN_AIN1, LOW);  digitalWrite(PIN_BIN1, HIGH); }  // CCW
  else      { digitalWrite(PIN_AIN1, HIGH); digitalWrite(PIN_BIN1, LOW);  }  // CW
  analogWrite(PIN_PWMA, TURN_SPEED);
  analogWrite(PIN_PWMB, TURN_SPEED);

  unsigned long d = (unsigned long)MS_FOR_90 * degrees / 90;   // scale from the calibrated 90
  turnEndTime = millis() + d;
  turning = true;

  Serial.print(left ? "Turning left " : "Turning right ");
  Serial.print(degrees); Serial.print(" deg over "); Serial.print(d); Serial.println(" ms");
}

void forward()   { digitalWrite(PIN_AIN1, HIGH); digitalWrite(PIN_BIN1, HIGH); analogWrite(PIN_PWMA, DRIVE_SPEED); analogWrite(PIN_PWMB, DRIVE_SPEED); }
void backward()  { digitalWrite(PIN_AIN1, LOW);  digitalWrite(PIN_BIN1, LOW);  analogWrite(PIN_PWMA, DRIVE_SPEED); analogWrite(PIN_PWMB, DRIVE_SPEED); }
void stopMotors(){ analogWrite(PIN_PWMA, 0); analogWrite(PIN_PWMB, 0); }
