#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <atomic>

using namespace std;
using namespace std::chrono;

void partialSum(const vector<int>& arr, int start, int end, long long &result) {
    long long localSum = 0;
    for (int i = start; i < end; i++)
        localSum += arr[i];
    result = localSum;
}

void partialSearch(const vector<int>& arr, int start, int end, int key, atomic<int> &result) {
    for (int i = start; i < end; i++) {
        if (arr[i] == key) {
            result = i;
            return;
        }
    }
}

int main() {
    int n, numThreads;
    cout << "Enter array size: ";
    cin >> n;

    cout << "Enter number of threads: ";
    cin >> numThreads;

    vector<int> arr(n);

    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    int key;
    cout << "Enter key to search: ";
    cin >> key;

    vector<thread> threads(numThreads);
    vector<long long> partialResults(numThreads);

    int blockSize = n / numThreads;

    // ----------- SUM USING THREADS -----------
    auto start = high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++) {
        int startIdx = i * blockSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + blockSize;

        threads[i] = thread(partialSum, cref(arr), startIdx, endIdx, ref(partialResults[i]));
    }

    for (int i = 0; i < numThreads; i++)
        threads[i].join();

    long long totalSum = 0;
    for (int i = 0; i < numThreads; i++)
        totalSum += partialResults[i];

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Threaded Sum: " << totalSum << endl;
    cout << "Time taken for Threaded Sum: " << duration.count() << " ms\n";

    // ----------- SEARCH USING THREADS -----------
    atomic<int> foundIndex(-1);

    start = high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++) {
        int startIdx = i * blockSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + blockSize;

        threads[i] = thread(partialSearch, cref(arr), startIdx, endIdx, key, ref(foundIndex));
    }

    for (int i = 0; i < numThreads; i++)
        threads[i].join();

    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    if (foundIndex != -1)
        cout << "Key found at index: " << foundIndex << endl;
    else
        cout << "Key not found\n";

    cout << "Time taken for Threaded Search: " << duration.count() << " ms\n";

    return 0;
}
