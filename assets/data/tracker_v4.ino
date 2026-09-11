// 双轴太阳能跟踪器 —— v4：带位置反馈
// ACEBOTT ESP32 Max V1.0 + 2×MG90S + 4×LDR + 2×电位器反馈 + SSD1306
//
// v4 新增：
//   从舵机内部电位器引出中心抽头，读到【真实角度】。
//   开机流程变成：读实际位置 → attach 时先写这个位置（舵机原地不动）
//   → 再用 S 曲线把指令推向 home。全程无猛冲，因为起点是测出来的。
//
//   这才是真正的 homing。之前所有软化手段本质上都是在猜起点。
//
// 接线：
//   AZ 舵机信号 → GPIO 5   (H1)      AZ 电位器抽头 → GPIO 32
//   EL 舵机信号 → GPIO 16  (H2)      EL 电位器抽头 → GPIO 33
//   LDR: LU=36 RU=39 LD=34 RD=35     OLED: SDA=21 SCL=22

#include <ESP32Servo.h>
#include <Wire.h>
#include <U8g2lib.h>

// ════════ 可调参数 ════════════════════════════════════
float KP_AZ    = 150.0f;   // 误差 → 角速度
float KP_EL    = 120.0f;
float DEADBAND = 0.030f;   // 光敏误差死区
float MAX_RATE = 40.0f;    // 最大角速度 度/秒
float DARK_TOTAL = 600.0f; // 天黑判据

// ════════ 引脚 ════════════════════════════════════════
const int PIN_AZ = 5;
const int PIN_EL = 16;
const int POT_AZ = 32;     // 方位电位器抽头
const int POT_EL = 33;     // 俯仰电位器抽头

const int PIN_LU = 36, PIN_RU = 39, PIN_LD = 34, PIN_RD = 35;

#define PIN_SDA 21
#define PIN_SCL 22

// ════════ 行程 ════════════════════════════════════════
const float AZ_MIN = 60,  AZ_MAX = 180, AZ_HOME = 120;
const float EL_MIN = 110, EL_MAX = 160, EL_HOME = 130;
const int   US_MIN = 500, US_MAX = 2400;

// ════════ 电位器三点标定（2026-08-24 实测）══════════════
// 三点分段线性：两点外推中间点偏差 AZ -4.0°、EL -2.5°，
// 超过 ±2°，所以用三点分两段插值。
const int AZ_R60 = 1333, AZ_R120 = 2370, AZ_R180 = 3556;
const int EL_R110 = 2208, EL_R130 = 2538, EL_R160 = 3151;

float azRawToDeg(int r) {
  if (r < AZ_R120) return  60.0f + (r - AZ_R60)  * 60.0f / (AZ_R120 - AZ_R60);
  else             return 120.0f + (r - AZ_R120) * 60.0f / (AZ_R180 - AZ_R120);
}

float elRawToDeg(int r) {
  if (r < EL_R130) return 110.0f + (r - EL_R110) * 20.0f / (EL_R130 - EL_R110);
  else             return 130.0f + (r - EL_R130) * 30.0f / (EL_R160 - EL_R130);
}

// ════════ 全局 ════════════════════════════════════════
Servo svAZ, svEL;
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, PIN_SCL, PIN_SDA);

float azCmd = AZ_HOME, elCmd = EL_HOME;   // 指令角度
float azAct = AZ_HOME, elAct = EL_HOME;   // 实测角度（滤波后）
float errAZ = 0, errEL = 0, total = 0;
float fLU = 0, fRU = 0, fLD = 0, fRD = 0;

float calLU = 1.00f, calRU = 1.00f, calLD = 1.00f, calRD = 1.00f;

enum State { HOMING, TRACKING, HOLDING, NIGHT };
State state = HOMING;
const char* stateName[] = { "HOMING", "TRACK", "HOLD", "NIGHT" };

unsigned long tCtrl = 0, tDraw = 0, tLog = 0;
const int CTRL_MS = 20;
const int DRAW_MS = 60;

// ════════ 工具 ════════════════════════════════════════
float sCurve(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }

int readMedian(int pin, int n = 5) {
  int v[9];
  n = constrain(n, 3, 9);
  for (int i = 0; i < n; i++) { v[i] = analogRead(pin); delayMicroseconds(150); }
  for (int i = 1; i < n; i++) {
    int k = v[i], j = i - 1;
    while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
    v[j + 1] = k;
  }
  return v[n / 2];
}

// 多次采样求平均，开机读起始位置时用
int readAverage(int pin, int n) {
  long s = 0;
  for (int i = 0; i < n; i++) { s += analogRead(pin); delay(3); }
  return s / n;
}

// ════════ 传感器 ══════════════════════════════════════
void sampleSensors() {
  const float A = 0.30f;
  fLU = fLU * (1 - A) + readMedian(PIN_LU) * A;
  fRU = fRU * (1 - A) + readMedian(PIN_RU) * A;
  fLD = fLD * (1 - A) + readMedian(PIN_LD) * A;
  fRD = fRD * (1 - A) + readMedian(PIN_RD) * A;

  float lu = fLU * calLU, ru = fRU * calRU;
  float ld = fLD * calLD, rd = fRD * calRD;

  total = lu + ru + ld + rd;
  float t = total + 1.0f;
  errAZ = ((lu + ld) - (ru + rd)) / t;
  errEL = ((lu + ru) - (ld + rd)) / t;
}

// 读实际角度（电位器），噪声较大所以滤得重一些
void samplePosition() {
  const float A = 0.15f;
  azAct = azAct * (1 - A) + azRawToDeg(readMedian(POT_AZ, 7)) * A;
  elAct = elAct * (1 - A) + elRawToDeg(readMedian(POT_EL, 7)) * A;
}

// ════════ 输出与控制 ══════════════════════════════════
void applyServos() {
  azCmd = constrain(azCmd, AZ_MIN, AZ_MAX);
  elCmd = constrain(elCmd, EL_MIN, EL_MAX);
  svAZ.write((int)roundf(azCmd));
  svEL.write((int)roundf(elCmd));
}

float shape(float e, float dz) {
  if (fabsf(e) <= dz) return 0.0f;
  return (e > 0) ? (e - dz) : (e + dz);
}

void controlStep(float dt) {
  float eA = shape(errAZ, DEADBAND);
  float eE = shape(errEL, DEADBAND);

  float rateAZ = constrain(KP_AZ * eA, -MAX_RATE, MAX_RATE);
  float rateEL = constrain(KP_EL * eE, -MAX_RATE, MAX_RATE);

  azCmd += rateAZ * dt;     // 本机实测方向
  elCmd += rateEL * dt;

  applyServos();
  state = (eA == 0 && eE == 0) ? HOLDING : TRACKING;
}

// 从当前指令角度平滑走到目标（起点已知，所以是真平滑）
void smoothMoveTo(float azTgt, float elTgt, int ms) {
  float a0 = azCmd, e0 = elCmd;
  int steps = ms / 20;
  for (int i = 0; i <= steps; i++) {
    float s = sCurve((float)i / steps);
    azCmd = a0 + (azTgt - a0) * s;
    elCmd = e0 + (elTgt - e0) * s;
    applyServos();
    samplePosition();
    delay(20);
  }
  azCmd = azTgt; elCmd = elTgt;
  applyServos();
}

// ════════ 显示 ════════════════════════════════════════
void centerBar(int x, int y, int w, float v) {
  u8g2.drawFrame(x, y, w, 7);
  int mid = x + w / 2;
  u8g2.drawVLine(mid, y - 2, 3);
  int p = mid + (int)(constrain(v, -1.f, 1.f) * (w / 2 - 3));
  u8g2.drawBox(p - 1, y + 1, 3, 5);
}

void draw() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x8_tr);

  u8g2.setCursor(0, 8);
  u8g2.printf("%-6s L:%4.0f", stateName[state], total);
  u8g2.drawHLine(0, 11, 128);

  // 指令 / 实测 / 差值
  u8g2.setCursor(0, 21);
  u8g2.printf("AZ %5.1f>%5.1f %+4.1f", azCmd, azAct, azAct - azCmd);
  u8g2.setCursor(0, 31);
  u8g2.printf("EL %5.1f>%5.1f %+4.1f", elCmd, elAct, elAct - elCmd);

  u8g2.drawHLine(0, 34, 128);

  // 光敏误差条
  u8g2.setCursor(0, 44);  u8g2.print("A");
  centerBar(12, 38, 116, errAZ * 4);
  u8g2.setCursor(0, 56);  u8g2.print("E");
  centerBar(12, 50, 116, errEL * 4);

  u8g2.setCursor(0, 64);
  u8g2.printf("%+5.1f%% %+5.1f%%", errAZ * 100, errEL * 100);

  u8g2.sendBuffer();
}

void drawHoming(float azS, float elS) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(0, 10, "HOMING");
  u8g2.drawHLine(0, 13, 128);
  u8g2.setFont(u8g2_font_5x8_tr);
  u8g2.setCursor(0, 26);
  u8g2.printf("found  AZ%5.1f EL%5.1f", azS, elS);
  u8g2.setCursor(0, 38);
  u8g2.printf("target AZ%5.1f EL%5.1f", AZ_HOME, EL_HOME);
  u8g2.setCursor(0, 54);
  u8g2.print("smooth transit...");
  u8g2.sendBuffer();
}

// ════════ 主程序 ══════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println("\n=== 太阳能跟踪器 v4（位置反馈版）===");

  analogSetAttenuation(ADC_11db);
  for (int p : {PIN_LU, PIN_RU, PIN_LD, PIN_RD, POT_AZ, POT_EL}) pinMode(p, INPUT);

  Wire.begin(PIN_SDA, PIN_SCL);
  u8g2.begin();
  u8g2.setContrast(180);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  svAZ.setPeriodHertz(50);
  svEL.setPeriodHertz(50);

  // ── 关键：先读实际位置，此时舵机还没 attach，输出轴自由 ──
  state = HOMING;
  int rawAZ = readAverage(POT_AZ, 40);
  int rawEL = readAverage(POT_EL, 40);
  float azStart = constrain(azRawToDeg(rawAZ), AZ_MIN, AZ_MAX);
  float elStart = constrain(elRawToDeg(rawEL), EL_MIN, EL_MAX);

  Serial.printf("开机实测位置: AZ %.1f (raw %d), EL %.1f (raw %d)\n",
                azStart, rawAZ, elStart, rawEL);
  drawHoming(azStart, elStart);

  azAct = azStart; elAct = elStart;

  // ── attach 后立刻写入实测角度：舵机收到「待在原地」──
  svAZ.attach(PIN_AZ, US_MIN, US_MAX);
  svAZ.write((int)roundf(azStart));
  svEL.attach(PIN_EL, US_MIN, US_MAX);
  svEL.write((int)roundf(elStart));
  azCmd = azStart; elCmd = elStart;
  delay(400);

  // ── 从真实起点平滑过渡到 home ──
  smoothMoveTo(AZ_HOME, EL_HOME, 1600);

  // 光敏滤波器初值
  fLU = analogRead(PIN_LU); fRU = analogRead(PIN_RU);
  fLD = analogRead(PIN_LD); fRD = analogRead(PIN_RD);

  state = TRACKING;
  Serial.println("ms,state,azCmd,azAct,elCmd,elAct,errAZ,errEL,total");
  tCtrl = tDraw = tLog = millis();
}

void loop() {
  unsigned long now = millis();

  if (now - tCtrl >= CTRL_MS) {
    float dt = (now - tCtrl) / 1000.0f;
    tCtrl = now;

    sampleSensors();
    samplePosition();

    if (total < DARK_TOTAL) {
      if (state != NIGHT) {
        state = NIGHT;
        smoothMoveTo(AZ_HOME, EL_HOME, 1500);
        tCtrl = millis();
      }
    } else {
      if (state == NIGHT) state = TRACKING;
      controlStep(dt);
    }
  }

  if (now - tDraw >= DRAW_MS) { tDraw = now; draw(); }

  if (now - tLog >= 500) {
    tLog = now;
    Serial.printf("%lu,%s,%.1f,%.1f,%.1f,%.1f,%.4f,%.4f,%.0f\n",
                  now, stateName[state], azCmd, azAct, elCmd, elAct,
                  errAZ, errEL, total);
  }
}

// ══════════════════════════════════════════════════════
// 屏幕读法：
//   AZ 120.0>118.3 -1.7
//      指令   实测   差值 ← 这就是舵机的稳态误差（死区+旷量）
//
// 现在能做而以前做不到的事：
//   · 开机不猛冲（起点是测出来的，不是猜的）
//   · 看到指令和实际的差 → 量化死区
//   · 串口日志多了 azAct/elAct 两列 → 可以画阶跃响应曲线
//
// 下一步可以考虑（不急）：
//   · 用实测角度做外环，纠正舵机自身的稳态误差
//   · 检测「指令变了但实测不动」→ 堵转或机构卡死报警
//   · AZ 噪声偏大（±3-5°），先确认是机械抖动还是电噪声：
//     用手扶住方位轴看噪声降不降
// ══════════════════════════════════════════════════════
