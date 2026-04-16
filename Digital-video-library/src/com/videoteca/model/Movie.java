package com.videoteca.model;

public class Movie extends MediaItem {
    private final int durationMinutes;

    public Movie(String id, String title, int year, double rating, int durationMinutes) {
        super(id, title, year, rating);
        this.durationMinutes = durationMinutes;
    }

    @Override
    public String getDetails() {
        return String.format(
            "Movie: %s (%d) - %d min (Rating: %.1f)",
            getTitle(), getYear(), durationMinutes, getRating()
        );
    }

    public int getDurationMinutes() {
        return durationMinutes;
    }
}