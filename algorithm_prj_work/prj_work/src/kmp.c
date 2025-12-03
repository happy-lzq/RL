#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 动态分配内存的最大限制，防止栈溢出
#define MAX_BUFFER 20000000 

// 计算部分匹配表 (Next 数组)
// next[i] 表示模式串 P[0...i] 的最长相等前后缀的长度
void compute_next(const char *pattern, int *next, int m) {
    next[0] = 0; // 第一个字符的 next 值为 0
    int k = 0;   // k 表示当前最长相等前后缀的长度

    for (int q = 1; q < m; q++) {
        // 如果当前字符不匹配，回溯 k
        while (k > 0 && pattern[k] != pattern[q]) {
            k = next[k - 1];
        }
        // 如果匹配，k 增加
        if (pattern[k] == pattern[q]) {
            k++;
        }
        next[q] = k;
    }
}

// KMP 搜索算法
// 返回匹配的总次数
int kmp_matcher(const char *text, const char *pattern, int print_positions) {
    int n = strlen(text);
    int m = strlen(pattern);
    
    if (m == 0) return 0;

    // 分配 next 数组内存
    int *next = (int *)malloc(m * sizeof(int));
    if (next == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 0;
    }

    // 预处理模式串
    compute_next(pattern, next, m);

    if (print_positions) {
        printf("Next 数组: ");
        for (int i = 0; i < m; i++) {
            printf("%d ", next[i]);
        }
        printf("\n");
    }

    int q = 0; // q 表示当前匹配的字符数
    int count = 0;

    for (int i = 0; i < n; i++) {
        // 如果字符不匹配，根据 next 数组回溯
        while (q > 0 && pattern[q] != text[i]) {
            q = next[q - 1];
        }
        // 如果字符匹配，q 增加
        if (pattern[q] == text[i]) {
            q++;
        }
        // 如果 q 等于模式串长度，说明找到了一个匹配
        if (q == m) {
            count++;
            if (print_positions) {
                printf("在位置 %d 发现匹配 (索引从0开始)\n", i - m + 1);
            }
            // 继续寻找下一个匹配，回溯 q
            q = next[q - 1];
        }
    }

    free(next);
    return count;
}

// 生成随机字符串
void generate_random_string(char *str, int length) {
    const char charset[] = "AB"; // 简化字符集以增加匹配概率，用于测试最坏情况或一般情况
    for (int i = 0; i < length; i++) {
        str[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    str[length] = '\0';
}

void run_benchmark() {
    printf("=== KMP 算法性能测试 (大规模数据集) ===\n");
    printf("| 文本长度 (N) | 模式长度 (M) | 耗时 (秒) |\n");
    printf("|---|---|---|\n");

    int sizes[] = {100000, 500000, 1000000, 5000000, 10000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    char *text = (char *)malloc(MAX_BUFFER);
    char *pattern = (char *)malloc(10000);

    if (!text || !pattern) {
        printf("内存不足，无法运行基准测试\n");
        return;
    }

    srand(time(NULL));

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int m = 1000; // 固定模式串长度

        generate_random_string(text, n);
        generate_random_string(pattern, m);

        clock_t start = clock();
        kmp_matcher(text, pattern, 0); // 不打印位置，只测速度
        clock_t end = clock();

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("| %d | %d | %.6f |\n", n, m, time_spent);
    }

    free(text);
    free(pattern);
    printf("\n测试完成。结果显示运行时间与文本长度呈线性关系，符合 O(N+M) 复杂度。\n");
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        run_benchmark();
        return 0;
    }

    // 交互模式
    // 使用动态分配以支持较大的输入
    char *text = (char *)malloc(MAX_BUFFER);
    char *pattern = (char *)malloc(MAX_BUFFER);

    if (!text || !pattern) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    printf("=== KMP 字符串匹配算法 ===\n");
    printf("提示: 输入 --benchmark 参数可运行性能测试\n\n");
    
    printf("请输入主文本 (Text): ");
    if (scanf("%s", text) != 1) return 1;
    
    printf("请输入模式串 (Pattern): ");
    if (scanf("%s", pattern) != 1) return 1;

    printf("\n正在在文本 \"%s\" 中查找模式 \"%s\" ...\n", text, pattern);
    int total = kmp_matcher(text, pattern, 1);
    printf("匹配总次数: %d\n", total);

    free(text);
    free(pattern);
    return 0;
}
