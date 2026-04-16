package com.videoteca.model;

public abstract class MediaItem {
    private final String id;
    private final String title;
    private final int year;
    private double rating;

    protected MediaItem(String id, String title, int year, double rating) {
        this.id = id;
        this.title = title;
        this.year = year;
        this.rating = rating;
    }

    /**
     * Returns detailed information about the media item.
     */
    public abstract String getDetails();

    // Getters and setters
    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public int getYear() {
        return year;
    }

    public double getRating() {
        return rating;
    }

    public void setRating(double rating) {
        this.rating = rating;
    }
}