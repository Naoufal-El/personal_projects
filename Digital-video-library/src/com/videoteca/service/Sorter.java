package com.videoteca.service;

import com.videoteca.model.MediaItem;
import java.util.Collections;
import java.util.List;

public class Sorter {
    public static void quickSortByTitle(List<MediaItem> list) {
        quickSort(list, 0, list.size() - 1);
    }

    private static void quickSort(List<MediaItem> a, int low, int high) {
        if (low < high) {
            int p = partition(a, low, high);
            quickSort(a, low, p - 1);
            quickSort(a, p + 1, high);
        }
    }

    private static int partition(List<MediaItem> a, int low, int high) {
        String pivot = a.get(high).getTitle();
        int i = low;
        for (int j = low; j < high; j++) {
            if (a.get(j).getTitle().compareToIgnoreCase(pivot) <= 0) {
                Collections.swap(a, i++, j);
            }
        }
        Collections.swap(a, i, high);
        return i;
    }

    public static void bubbleSortByYear(List<MediaItem> list) {
        int n = list.size();
        boolean swapped;
        for (int i = 0; i < n - 1; i++) {
            swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (list.get(j).getYear() > list.get(j+1).getYear()) {
                    Collections.swap(list, j, j+1);
                    swapped = true;
                }
            }
            if (!swapped) break;
        }
    }

    public static void bubbleSortByRating(List<MediaItem> list) {
        int n = list.size();
        boolean swapped;
        for (int i = 0; i < n - 1; i++) {
            swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (list.get(j).getRating() > list.get(j+1).getRating()) {
                    Collections.swap(list, j, j+1);
                    swapped = true;
                }
            }
            if (!swapped) break;
        }
    }
}
