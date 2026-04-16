package com.videoteca.service;

import com.videoteca.model.Rental;
import com.videoteca.model.User;
import com.videoteca.persistence.RentalDAO;
import com.videoteca.persistence.UserDAO;
import com.videoteca.persistence.SQLiteDataSource;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

public class RentalService {
    private final RentalDAO rentalDAO;
    private final UserDAO userDAO;
    private final Map<String, Rental> rentals = new LinkedHashMap<>();

    public RentalService(SQLiteDataSource ds) {
        this.rentalDAO = new RentalDAO(ds);
        this.userDAO   = new UserDAO(ds);
    }

    public void loadAll() throws SQLException {
        List<Rental> list = rentalDAO.findAll();
        rentals.clear();
        for (Rental r : list) {
            rentals.put(r.getRentalId(), r);
        }
    }

    public void saveAll() throws SQLException {
        for (Rental r : rentals.values()) {
            if (rentalDAO.findById(r.getRentalId()) != null) {
                rentalDAO.update(r);
            } else {
                rentalDAO.insert(r);
            }
        }
    }

    /**
     * Requests a new rental; persists immediately.
     */
    public void requestRental(String userId, String mediaId) throws SQLException {
        User user = userDAO.findById(userId);
        if (user == null) {
            throw new IllegalArgumentException("User not found: " + userId);
        }
        int active = activeRentalsCount(userId);
        if (!user.canRentMore(active)) {
            throw new IllegalStateException("User has reached rental limit");
        }
        // Generate a new unique rentalId
        String rentalId;
        int maxNum = 0;
        for (String rid : rentals.keySet()) {
            if (rid.startsWith("noleggio")) {
                try {
                    int num = Integer.parseInt(rid.replaceAll("\\D", ""));
                    if (num > maxNum) {
                        maxNum = num;
                    }
                } catch (NumberFormatException e) {
                    // ignore
                }
            }
        }
        rentalId = "noleggio" + (maxNum + 1);
        // Prevent same user renting the same item twice concurrently
        if (rentals.values().stream().anyMatch(r -> r.getUserId().equals(userId)
                                                  && r.getMediaId().equals(mediaId)
                                                  && r.getReturnDate() == null)) {
            throw new IllegalArgumentException("User already has an active rental for media " + mediaId);
        }
        Rental r = new Rental(rentalId, userId, mediaId, LocalDate.now());
        rentals.put(rentalId, r);
        rentalDAO.insert(r);  // immediately persist
    }

    /**
     * Returns an active rental by media ID for a user; persists immediately.
     */
    public void returnByMedia(String userId, String mediaId) throws SQLException {
        Optional<Rental> match = rentals.values().stream()
            .filter(r -> r.getUserId().equals(userId)
                      && r.getMediaId().equals(mediaId)
                      && r.getReturnDate() == null)
            .findFirst();
        if (match.isPresent()) {
            Rental r = match.get();
            r.setReturnDate(LocalDate.now());
            rentalDAO.update(r); // persist return date
        } else {
            throw new IllegalArgumentException(
                "No active rental found for media ID: " + mediaId);
        }
    }

    public int activeRentalsCount(String userId) {
        return (int) rentals.values().stream()
            .filter(r -> r.getUserId().equals(userId) && r.getReturnDate() == null)
            .count();
    }

    public List<Rental> listAll(Optional<String> maybeUserId) {
        if (maybeUserId.isPresent()) {
            String uid = maybeUserId.get();
            return rentals.values().stream()
                   .filter(r -> r.getUserId().equals(uid))
                   .collect(Collectors.toList());
        }
        return new ArrayList<>(rentals.values());
    }

    /**
     * Removes a rental record by ID (admin operation).
     */
    public void remove(String rentalId) throws SQLException {
        Rental removed = rentals.remove(rentalId);
        if (removed == null) {
            throw new IllegalArgumentException("Rental with ID '" + rentalId + "' does not exist.");
        }
        rentalDAO.delete(rentalId);
    }
}
