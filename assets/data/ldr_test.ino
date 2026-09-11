// ⚠️ 版本说明（后加）
// 这是早期的四象限光敏测试程序，那时还没有电位器反馈，
// 光敏占用 GPIO 32/33/34/35。
// 加入舵机位置反馈之后，32/33 让给了电位器抽头，
// 光敏改到 GPIO 36/39/34/35（见 tracker_v4.ino）。
// 单独跑这个测试程序时按本文件的引脚接线即可，
// 但要和 v4 一起用，请先把下面的引脚改成 36/39/34/35。

// 太阳能跟踪器 —— 四象限光敏读数测试（只读，不驱动舵机）
// ACEBOTT ESP32 Max V1.0 + 4×LDR + SSD1306
//
// 目的：
//   1. 验证四路接线是否正确（遮哪个，哪个掉）
//   2. 看四个传感器的一致性差多少（决定要不要标定）
//   3. 测 ADC 噪声有多大（决定死区和滤波强度）
//   4. 看光照变化的动态范围
//
// 用法：
//   烧进去，屏上四个数按物理位置排布。用手依次遮挡每个象限，
//   确认对应的数字下降。串口每秒打印一行，方便复制到表格里分析。
//
// 需要的库：U8g2

#include <Wire.h>
#include <U8g2lib.h>

// ── 引脚（全部 ADC1，开 WiFi 也能用）─────────────────
const int PIN_LU = 34;   // 左上
const int PIN_RU = 35;   // 右上
const int PIN_LD = 32;   // 左下
const int PIN_RD = 33;   // 右下

#define PIN_SDA 21
#define PIN_SCL 22

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, PIN_SCL, PIN_SDA);

// ── 采样与滤波 ────────────────────────
const int OVERSAMPLE = 8;      // 每次读取的连采次数，取中位数
const float EMA_A = 0.25f;     // 一阶低通系数，越小越平滑越滞后

struct Ch {
  int pin;
  const char* name;
  int raw;                     // 本次原始值
  float filt;                  // 滤波后
  int lo, hi;                  // 观察到的最小/最大，用来看动态范围
  float cal;                   // 标定系数，先都设 1.0
};

Ch ch[4] = {
  { PIN_LU, "LU", 0, 0, 4095, 0, 1.0f },
  { PIN_RU, "RU", 0, 0, 4095, 0, 1.0f },
  { PIN_LD, "LD", 0, 0, 4095, 0, 1.0f },
  { PIN_RD, "RD", 0, 0, 4095, 0, 1.0f },
};
#define LU 0
#define RU 1
#define LD 2
#define RD 3

// 噪声统计：记录最近若干次读数的波动范围
const int NBUF = 32;
int nbuf[4][NBUF];
int nidx = 0;
int noise[4] = {0, 0, 0, 0};

// 中位数采样：对孤立尖峰免疫，比取平均可靠
int readMedian(int pin) {
  int v[OVERSAMPLE];
  for (int i = 0; i < OVERSAMPLE; i++) {
    v[i] = analogRead(pin);
    delayMicroseconds(200);
  }
  // 小数组，插入排序足够
  for (int i = 1; i < OVERSAMPLE; i++) {
    int k = v[i], j = i - 1;
    while (j >= 0 && v[j] > k) { v[j + 1] = v[j]; j--; }
    v[j + 1] = k;
  }
  return (v[OVERSAMPLE / 2 - 1] + v[OVERSAMPLE / 2]) / 2;
}

void sampleAll() {
  for (int i = 0; i < 4; i++) {
    ch[i].raw = readMedian(ch[i].pin);
    ch[i].filt = ch[i].filt * (1 - EMA_A) + ch[i].raw * EMA_A;

    if (ch[i].raw < ch[i].lo) ch[i].lo = ch[i].raw;
    if (ch[i].raw > ch[i].hi) ch[i].hi = ch[i].raw;

    nbuf[i][nidx] = ch[i].raw;
  }
  nidx = (nidx + 1) % NBUF;

  // 每轮算一次噪声峰峰值
  for (int i = 0; i < 4; i++) {
    int mn = 4095, mx = 0;
    for (int k = 0; k < NBUF; k++) {
      if (nbuf[i][k] < mn) mn = nbuf[i][k];
      if (nbuf[i][k] > mx) mx = nbuf[i][k];
    }
    noise[i] = mx - mn;
  }
}

// ── 误差计算 ────────────────────────────
// 归一化：除以总光强，这样阴天晴天可以用同一组参数
float errAZ, errEL, total;

void computeError() {
  float lu = ch[LU].filt * ch[LU].cal;
  float ru = ch[RU].filt * ch[RU].cal;
  float ld = ch[LD].filt * ch[LD].cal;
  float rd = ch[RD].filt * ch[RD].cal;

  total = lu + ru + ld + rd + 1.0f;         // +1 防除零

  // 注意：光敏电阻模块多数是"越亮读数越小"，如果你的相反，
  // 把下面两行的正负对调即可（跑一下就知道）
  errAZ = ((lu + ld) - (ru + rd)) / total;  // 正 = 左边亮
  errEL = ((lu + ru) - (ld + rd)) / total;  // 正 = 上边亮
}

// ── 显示 ──────────────────────────────
// 居中的双向误差条：指针在中间表示对准
void centerBar(int x, int y, int w, float v) {
  u8g2.drawFrame(x, y, w, 7);
  int mid = x + w / 2;
  u8g2.drawVLine(mid, y - 2, 3);                 // 中心刻度
  int p = mid + (int)(constrain(v, -1.0f, 1.0f) * (w / 2 - 3));
  u8g2.drawBox(p - 1, y + 1, 3, 5);              // 指针
}

void draw() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x8_tr);

  // 四个数按物理位置排布，和实物一一对应
  u8g2.setCursor(6, 8);    u8g2.printf("LU%4d", ch[LU].raw);
  u8g2.setCursor(70, 8);   u8g2.printf("RU%4d", ch[RU].raw);
  u8g2.setCursor(6, 20);   u8g2.printf("LD%4d", ch[LD].raw);
  u8g2.setCursor(70, 20);  u8g2.printf("RD%4d", ch[RD].raw);
  u8g2.drawVLine(64, 0, 24);
  u8g2.drawHLine(0, 26, 128);

  // 两条误差条
  u8g2.setCursor(0, 38);   u8g2.print("AZ");
  centerBar(20, 32, 106, errAZ * 4);              // ×4 放大，小误差也看得见

  u8g2.setCursor(0, 50);   u8g2.print("EL");
  centerBar(20, 44, 106, errEL * 4);

  // 底部：误差百分比 + 最大噪声
  int mxNoise = 0;
  for (int i = 0; i < 4; i++) if (noise[i] > mxNoise) mxNoise = noise[i];
  u8g2.setCursor(0, 62);
  u8g2.printf("A%+5.1f%% E%+5.1f%% n%d", errAZ * 100, errEL * 100, mxNoise);

  u8g2.sendBuffer();
}

// ── 主程序 ─────────────────────────────
unsigned long lastPrint = 0;

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n=== 四象限光敏读数测试 ===");
  Serial.println("遮挡各象限，确认对应数值变化");
  Serial.println("ms,LU,RU,LD,RD,errAZ,errEL,total");

  // 11dB 衰减 = 量程约 0-3.3V（但两端非线性，见下面提示）
  analogSetAttenuation(ADC_11db);

  for (int i = 0; i < 4; i++) {
    pinMode(ch[i].pin, INPUT);
    ch[i].filt = analogRead(ch[i].pin);
    for (int k = 0; k < NBUF; k++) nbuf[i][k] = ch[i].filt;
  }

  Wire.begin(PIN_SDA, PIN_SCL);
  u8g2.begin();
  u8g2.setContrast(180);
}

void loop() {
  sampleAll();
  computeError();
  draw();

  // 每 500ms 打印一行，方便复制到表格分析
  if (millis() - lastPrint > 500) {
    lastPrint = millis();
    Serial.printf("%lu,%d,%d,%d,%d,%.4f,%.4f,%.0f\n",
                  millis(), ch[LU].raw, ch[RU].raw, ch[LD].raw, ch[RD].raw,
                  errAZ, errEL, total);
  }

  delay(30);
}

// ─────────────────────────────────
// 看完读数之后，记下这几个数字，后面调参要用：
//
//   1. 四个通道全遮暗时的读数        → 暗电平
//   2. 四个通道均匀照亮时的读数      → 差异越大越需要标定
//      标定方法：取四个的平均值 M，把每个通道的 cal 设成 M / 该通道读数
//   3. 屏幕右下角那个 n（噪声峰峰值） → 死区至少要设成它的 2 倍
//   4. errAZ / errEL 在完全对准时是否接近 0
//
// 如果遮左边反而是右边的数变小，说明接线顺序反了，改引脚定义即可。
// 如果遮挡时数值上升而不是下降，说明模块是"越亮读数越小"，
// 把 computeError() 里两行的正负对调。
// ──────────────────────────────────
