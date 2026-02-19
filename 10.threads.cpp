#include <stdio.h>
#include <pthread.h>

void* printNumbers(void* arg) {
    int n = *(int*)arg;
    for(int i = 1; i <= n; i++) {
        printf("%d ", i);
    }
    printf("\n");
    return NULL;
}

int main() {
    pthread_t thread;
    int n;

    printf("Enter value of n: ");
    scanf("%d", &n);

    pthread_create(&thread, NULL, printNumbers, &n);
    pthread_join(thread, NULL);

    return 0;
}
