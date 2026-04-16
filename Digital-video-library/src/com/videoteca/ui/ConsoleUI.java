package com.videoteca.ui;

import com.videoteca.model.AdminUser;
import com.videoteca.model.PremiumUser;
import com.videoteca.model.RegularUser;
import com.videoteca.model.User;
import com.videoteca.model.MediaItem;
import com.videoteca.model.Rental;
import com.videoteca.persistence.UserDAO;
import com.videoteca.service.CatalogService;
import com.videoteca.service.RentalService;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Scanner;

public class ConsoleUI {
    private final CatalogService catalog;
    private final RentalService rentalService;
    private final Scanner scanner = new Scanner(System.in);
    private User currentUser;

    public ConsoleUI(CatalogService catalog, RentalService rentalService) {
        this.catalog = catalog;
        this.rentalService = rentalService;
    }

    public void start() {
        login();
        while (true) {
            showMenu();
            String choice = scanner.nextLine().trim();
            try {
                switch (choice) {
                    case "1": listMedia(); break;
                    case "2": searchMedia(); break;
                    case "3": requestRental(); break;
                    case "4": returnRental(); break;
                    case "5": listRentals(); break;
                    case "6": addMediaItem(); break;
                    case "7": removeMediaItem(); break;
                    case "8": addUser(); break;
                    case "9": removeUser(); break;
                    case "10": removeRental(); break;
                    case "0": return;
                    default: System.out.println("Invalid choice");
                }
            } catch (SQLException | IllegalStateException | IllegalArgumentException e) {
                System.out.println("Error: " + e.getMessage());
            }
        }
    }

    private void login() {
        System.out.print("Enter your user ID: ");
        String userId = scanner.nextLine().trim();
        try {
            UserDAO userDAO = new UserDAO(new com.videoteca.persistence.SQLiteDataSource());
            currentUser = userDAO.findById(userId);
            if (currentUser == null) {
                System.out.println("User not found. Exiting.");
                System.exit(1);
            }
        } catch (SQLException e) {
            System.out.println("Failed to load user: " + e.getMessage());
            System.exit(1);
        }
        System.out.println("Welcome, " + currentUser.getName() + "!\n");
    }

    private void showMenu() {
        System.out.println("=== Main Menu ===");
        System.out.println("1) List all media");
        System.out.println("2) Search media");
        System.out.println("3) Request rental");
        System.out.println("4) Return rental");
        System.out.println("5) List rentals");
        if (currentUser instanceof AdminUser) {
            System.out.println("6) Add media item");
            System.out.println("7) Remove media item");
            System.out.println("8) Add user");
            System.out.println("9) Remove user");
            System.out.println("10) Remove rental");
        }
        System.out.println("0) Exit");
        System.out.print("Choice: ");
    }

    private void listMedia() {
        List<MediaItem> items = new ArrayList<>(catalog.listAll());
        System.out.print("Sort by (title/year/rating) [default=title]: ");
        String sort = scanner.nextLine().trim().toLowerCase();
        if (sort.equals("year")) {
            com.videoteca.service.Sorter.bubbleSortByYear(items);
        } else if (sort.equals("rating")) {
            com.videoteca.service.Sorter.bubbleSortByRating(items);
        } else {
            com.videoteca.service.Sorter.quickSortByTitle(items);
        }
        items.forEach(m -> System.out.printf("Media ID: %s | %s%n", m.getId(), m.getDetails()));
        System.out.println();
    }

    private void searchMedia() {
        System.out.print("Enter title query: ");
        String q = scanner.nextLine().trim();
        List<MediaItem> results = catalog.searchByTitle(q);
        com.videoteca.service.Sorter.quickSortByTitle(results);
        if (results.isEmpty()) {
            System.out.println("No media found matching '" + q + "'.\n");
        } else {
            results.forEach(m -> System.out.printf("Media ID: %s | %s%n", m.getId(), m.getDetails()));
            System.out.println();
        }
    }

    private void requestRental() throws SQLException {
        System.out.print("Enter media ID to rent: ");
        String mediaId = scanner.nextLine().trim();
        MediaItem item = catalog.findById(mediaId);
        if (item == null) {
            System.out.println("Media item with ID '" + mediaId + "' does not exist.\n");
            return;
        }
        rentalService.requestRental(currentUser.getUserId(), mediaId);
        System.out.println("Rental requested successfully.\n");
    }

    private void returnRental() throws SQLException {
        System.out.print("Enter media ID to return: ");
        String mediaId = scanner.nextLine().trim();
        rentalService.returnByMedia(currentUser.getUserId(), mediaId);
        System.out.println("Return registered.\n");
    }

    private void listRentals() {
        List<Rental> rentals;
        if (currentUser instanceof AdminUser) {
            System.out.print("List rentals for (leave blank for all): ");
            String uid = scanner.nextLine().trim();
            Optional<String> maybeUser = uid.isEmpty() ? Optional.empty() : Optional.of(uid);
            rentals = rentalService.listAll(maybeUser);
        } else {
            rentals = rentalService.listAll(Optional.of(currentUser.getUserId()));
        }
        if (rentals.isEmpty()) {
            System.out.println("No rentals found." + (currentUser instanceof AdminUser ? "" : " for your user.") + "\n");
        } else {
            if (currentUser instanceof AdminUser) {
                rentals.forEach(r -> System.out.printf(
                    "Rental ID: %s | User: %s | Media ID: %s | Rented on: %s%s%n",
                    r.getRentalId(), r.getUserId(), r.getMediaId(), r.getRentalDate(),
                    r.getReturnDate() != null ? " | Returned on: " + r.getReturnDate() : ""
                ));
            } else {
                rentals.forEach(r -> System.out.printf(
                    "Media ID: %s | Rented on: %s%s%n",
                    r.getMediaId(), r.getRentalDate(),
                    r.getReturnDate() != null ? " | Returned on: " + r.getReturnDate() : ""
                ));
            }
            System.out.println();
        }
    }

    private void addMediaItem() throws SQLException {
        if (!(currentUser instanceof AdminUser)) {
            System.out.println("Unauthorized.");
            return;
        }
        System.out.print("Enter media ID: ");
        String id = scanner.nextLine().trim();
        System.out.print("Title: ");
        String title = scanner.nextLine().trim();
        System.out.print("Year: ");
        int year = Integer.parseInt(scanner.nextLine().trim());
        System.out.print("Rating: ");
        double rating = Double.parseDouble(scanner.nextLine().trim());
        System.out.print("Type (movie/tvseries): ");
        String type = scanner.nextLine().trim().toLowerCase();
        MediaItem item;
        if ("movie".equals(type)) {
            System.out.print("Duration (min): ");
            int duration = Integer.parseInt(scanner.nextLine().trim());
            item = new com.videoteca.model.Movie(id, title, year, rating, duration);
        } else {
            System.out.print("Seasons: ");
            int seasons = Integer.parseInt(scanner.nextLine().trim());
            System.out.print("Episodes: ");
            int episodes = Integer.parseInt(scanner.nextLine().trim());
            item = new com.videoteca.model.TVSeries(id, title, year, rating, seasons, episodes);
        }
        catalog.add(item);
        System.out.println("Media item added.\n");
    }

    private void removeMediaItem() throws SQLException {
        if (!(currentUser instanceof AdminUser)) {
            System.out.println("Unauthorized.");
            return;
        }
        System.out.print("Enter media ID to remove: ");
        String id = scanner.nextLine().trim();
        MediaItem item = catalog.findById(id);
        if (item == null) {
            System.out.println("Media item with ID '" + id + "' does not exist.\n");
        } else {
            catalog.remove(id);
            System.out.println("Media item removed.\n");
        }
    }

    private void addUser() throws SQLException {
        if (!(currentUser instanceof AdminUser)) {
            System.out.println("Unauthorized.");
            return;
        }
        UserDAO userDAO = new UserDAO(new com.videoteca.persistence.SQLiteDataSource());
        System.out.print("Enter name: ");
        String name = scanner.nextLine().trim();
        System.out.print("Enter role (regular/premium): ");
        String role = scanner.nextLine().trim().toLowerCase();
        User newUser;
        switch (role) {
            case "regular": newUser = new RegularUser(name); break;
            case "premium": newUser = new PremiumUser(name); break;
            default:
                System.out.println("Invalid role.\n");
                return;
        }
        String id = userDAO.insert(newUser);
        newUser.setUserId(id);
        System.out.println("User added with ID: " + id + " and role: " + role + ".\n");
    }

    private void removeUser() throws SQLException {
        if (!(currentUser instanceof AdminUser)) {
            System.out.println("Unauthorized.");
            return;
        }
        System.out.print("Enter user ID to remove: ");
        String idStr = scanner.nextLine().trim();
        UserDAO userDAO = new UserDAO(new com.videoteca.persistence.SQLiteDataSource());
        User u = userDAO.findById(idStr);
        if (u == null) {
            System.out.println("User with ID '" + idStr + "' does not exist.\n");
        } else {
            userDAO.delete(idStr);
            System.out.println("User removed.\n");
        }
    }

    private void removeRental() throws SQLException {
        System.out.print("Enter rental ID to remove: ");
        String id = scanner.nextLine().trim();
        try {
            rentalService.remove(id);
            System.out.println("Rental removed.\n");
        } catch (IllegalArgumentException e) {
            System.out.println("Rental with ID '" + id + "' does not exist.\n");
        }
    }
}
