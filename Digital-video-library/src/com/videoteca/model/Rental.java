package com.videoteca.model;

import java.time.LocalDate;

public class Rental {
    private final String rentalId;
    private final String userId;
    private final String mediaId;
    private final LocalDate rentalDate;
    private LocalDate returnDate;

    public Rental(String rentalId, String userId, String mediaId, LocalDate rentalDate) {
        this.rentalId = rentalId;
        this.userId = userId;
        this.mediaId = mediaId;
        this.rentalDate = rentalDate;
    }

    public String getRentalId() {
        return rentalId;
    }

    public String getUserId() {
        return userId;
    }

    public String getMediaId() {
        return mediaId;
    }

    public LocalDate getRentalDate() {
        return rentalDate;
    }

    public LocalDate getReturnDate() {
        return returnDate;
    }

    public void setReturnDate(LocalDate returnDate) {
        this.returnDate = returnDate;
    }
}
