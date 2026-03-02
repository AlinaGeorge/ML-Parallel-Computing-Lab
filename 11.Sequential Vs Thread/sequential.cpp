#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to find sum of array
long long findSum(const vector<int>& arr) {
    long long sum = 0;
    for (size_t i = 0; i < arr.size(); i++)
        sum += arr[i];
    return sum;
}

// Function to search key
int searchKey(const vector<int>& arr, int key) {
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == key)
            return i;  // return index if found
    }
    return -1; // not found
}

int main() {
    int n;
    cout << "Enter array size: ";
    cin >> n;

    vector<int> arr(n);

    // Generate random numbers
    srand(time(0));
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    int key;
    cout << "Enter key to search: ";
    cin >> key;

    // Measure sum time
    auto start = high_resolution_clock::now();
    long long sum = findSum(arr);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Sequential Sum: " << sum << endl;
    cout << "Time taken for Sum: " << duration.count() << " ms\n";

    // Measure search time
    start = high_resolution_clock::now();
    int index = searchKey(arr, key);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    if (index != -1)
        cout << "Key found at index: " << index << endl;
    else
        cout << "Key not found\n";

    cout << "Time taken for Search: " << duration.count() << " ms\n";

    return 0;
}
