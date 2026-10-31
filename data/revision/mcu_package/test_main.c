/* Host-side check of the headers: runs the 100 test windows through float32, int8 and int16 MLP
   pipelines and the float32 SNN, and compares with the expected values.  gcc -O2 -o test_main test_main.c -lm */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "mlp_weights_f32.h"
#include "mlp_weights_int8.h"
#include "mlp_weights_int16.h"
#include "mlp_test_vectors.h"
#include "snn_weights_f32.h"
#include "snn_test_vectors.h"

static float rha(float v) { return (v >= 0) ? floorf(v + 0.5f) : -floorf(-v + 0.5f); }
static long clampl(long v, long lo, long hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void mlp_f32(const float *x, float *y) {
    float h1[MLP_N_HID], h2[MLP_N_HID];
    for (int j = 0; j < MLP_N_HID; j++) { float a = MLP_b1[j]; for (int i = 0; i < MLP_N_IN; i++) a += MLP_W1[j][i] * x[i]; h1[j] = a > 0 ? a : 0; }
    for (int j = 0; j < MLP_N_HID; j++) { float a = MLP_b2[j]; for (int i = 0; i < MLP_N_HID; i++) a += MLP_W2[j][i] * h1[i]; h2[j] = a > 0 ? a : 0; }
    for (int j = 0; j < MLP_N_OUT; j++) { float a = MLP_b3[j]; for (int i = 0; i < MLP_N_HID; i++) a += MLP_W3[j][i] * h2[i]; y[j] = a; }
}
static void mlp_i8(const int8_t *x, float *y) {
    int32_t h1[MLP_N_HID], h2[MLP_N_HID];
    for (int j = 0; j < MLP_N_HID; j++) { int32_t a = MLP8_b1q[j]; for (int i = 0; i < MLP_N_IN; i++) a += (int32_t)MLP8_W1q[j][i] * x[i]; h1[j] = (int32_t)clampl((long)rha((a > 0 ? a : 0) * MLP8_M1), -128, 127); }
    for (int j = 0; j < MLP_N_HID; j++) { int32_t a = MLP8_b2q[j]; for (int i = 0; i < MLP_N_HID; i++) a += (int32_t)MLP8_W2q[j][i] * h1[i]; h2[j] = (int32_t)clampl((long)rha((a > 0 ? a : 0) * MLP8_M2), -128, 127); }
    for (int j = 0; j < MLP_N_OUT; j++) { int32_t a = MLP8_b3q[j]; for (int i = 0; i < MLP_N_HID; i++) a += (int32_t)MLP8_W3q[j][i] * h2[i]; y[j] = a * MLP8_M3; }
}
static void mlp_i16(const int16_t *x, float *y) {
    int64_t h1[MLP_N_HID], h2[MLP_N_HID];
    for (int j = 0; j < MLP_N_HID; j++) { int64_t a = MLP16_b1q[j]; for (int i = 0; i < MLP_N_IN; i++) a += (int64_t)MLP16_W1q[j][i] * x[i]; h1[j] = clampl((long)rha((double)(a > 0 ? a : 0) * MLP16_M1), -32768, 32767); }
    for (int j = 0; j < MLP_N_HID; j++) { int64_t a = MLP16_b2q[j]; for (int i = 0; i < MLP_N_HID; i++) a += (int64_t)MLP16_W2q[j][i] * h1[i]; h2[j] = clampl((long)rha((double)(a > 0 ? a : 0) * MLP16_M2), -32768, 32767); }
    for (int j = 0; j < MLP_N_OUT; j++) { int64_t a = MLP16_b3q[j]; for (int i = 0; i < MLP_N_HID; i++) a += (int64_t)MLP16_W3q[j][i] * h2[i]; y[j] = (float)((double)a * MLP16_M3); }
}
static void snn_f32(const uint8_t x16c[SNN_N_STEPS][SNN_N_IN], float *q, int *nspk) {
    float is1[SNN_N_HID] = {0}, v1[SNN_N_HID] = {0}, is2[SNN_N_HID] = {0}, v2[SNN_N_HID] = {0}, is3[SNN_N_OUT] = {0}, v3[SNN_N_OUT] = {0};
    float s1[SNN_N_HID], s2[SNN_N_HID];
    for (int j = 0; j < SNN_N_OUT; j++) q[j] = 0; *nspk = 0;
    for (int t = 0; t < SNN_N_STEPS; t++) {
        for (int j = 0; j < SNN_N_HID; j++) { float a = 0; for (int i = 0; i < SNN_N_IN; i++) a += 0.2f * x16c[t][i] * SNN_W0[j][i]; is1[j] = (is1[j] + a) * SNN_BETA; v1[j] = SNN_ALPHA * v1[j] + is1[j]; s1[j] = v1[j] >= 1.0f ? floorf(v1[j]) : 0.0f; v1[j] -= s1[j]; *nspk += (int)s1[j]; }
        for (int j = 0; j < SNN_N_HID; j++) { float a = 0; for (int i = 0; i < SNN_N_HID; i++) if (s1[i] != 0) a += s1[i] * SNN_W2[j][i]; is2[j] = (is2[j] + a) * SNN_BETA; v2[j] = SNN_ALPHA * v2[j] + is2[j]; s2[j] = v2[j] >= 1.0f ? floorf(v2[j]) : 0.0f; v2[j] -= s2[j]; *nspk += (int)s2[j]; }
        for (int j = 0; j < SNN_N_OUT; j++) { float a = 0; for (int i = 0; i < SNN_N_HID; i++) if (s2[i] != 0) a += s2[i] * SNN_W4[j][i]; is3[j] = (is3[j] + a) * SNN_BETA; v3[j] = SNN_ALPHA * v3[j] + is3[j]; q[j] += v3[j]; }
    }
}
static int act_ok(const float *y, const int8_t *exp) { for (int h = 0; h < 3; h++) { int b = 0; for (int k = 1; k < 3; k++) if (y[3*h+k] > y[3*h+b]) b = k; if (b != exp[h]) return 0; } return 1; }
int main(void) {
    float y[9]; int ok_f = 0, ok_8 = 0, ok_16 = 0, ok_s = 0; float e_f = 0, e_8 = 0, e_16 = 0, e_s = 0;
    for (int n = 0; n < MLP_N_TEST; n++) {
        mlp_f32(MLP_X[n], y); for (int j = 0; j < 9; j++) e_f = fmaxf(e_f, fabsf(y[j] - MLP_Y_F32[n][j])); ok_f += act_ok(y, MLP_ACT_F32[n]);
        mlp_i8(MLP_Xq8[n], y); for (int j = 0; j < 9; j++) e_8 = fmaxf(e_8, fabsf(y[j] - MLP_Y_INT8[n][j])); ok_8 += act_ok(y, MLP_ACT_INT8[n]);
        mlp_i16(MLP_Xq16[n], y); for (int j = 0; j < 9; j++) e_16 = fmaxf(e_16, fabsf(y[j] - MLP_Y_INT16[n][j])); ok_16 += act_ok(y, MLP_ACT_INT16[n]);
    }
    int sp; float q[9]; int spk_mismatch = 0;
    for (int n = 0; n < SNN_N_TEST; n++) { snn_f32(SNN_X16c[n], q, &sp); for (int j = 0; j < 9; j++) e_s = fmaxf(e_s, fabsf(q[j] - SNN_Q[n][j])); ok_s += act_ok(q, SNN_ACT[n]); spk_mismatch += (sp != SNN_SPIKES[n]); }
    printf("MLP float32: actions match %d/%d, max |dy| %.2e\n", ok_f, MLP_N_TEST, e_f);
    printf("MLP int8   : actions match %d/%d, max |dy| %.2e\n", ok_8, MLP_N_TEST, e_8);
    printf("MLP int16  : actions match %d/%d, max |dy| %.2e\n", ok_16, MLP_N_TEST, e_16);
    printf("SNN float32: actions match %d/%d, max |dQ| %.2e, spike-count mismatches %d\n", ok_s, SNN_N_TEST, e_s, spk_mismatch);
    return 0;
}
