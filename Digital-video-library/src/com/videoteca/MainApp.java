package com.videoteca;

import com.videoteca.persistence.SQLiteDataSource;
import com.videoteca.service.CatalogService;
import com.videoteca.service.RentalService;
import com.videoteca.ui.ConsoleUI;
import java.sql.SQLException;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainApp {
    public static void main(String[] args) {
        // Initialize the data source
        SQLiteDataSource ds = new SQLiteDataSource();

        // Initialize services
        CatalogService catalog = new CatalogService(ds);
        RentalService rentalService = new RentalService(ds);

        // Load data into memory concurrently
        AtomicBoolean loadError = new AtomicBoolean(false);
        Thread t1 = new Thread(() -> {
            try {
                catalog.loadAll();
            } catch (SQLException e) {
                System.err.println("Error loading catalog data: " + e.getMessage());
                e.printStackTrace();
                loadError.set(true);
            }
        });
        Thread t2 = new Thread(() -> {
            try {
                rentalService.loadAll();
            } catch (SQLException e) {
                System.err.println("Error loading rentals data: " + e.getMessage());
                e.printStackTrace();
                loadError.set(true);
            }
        });
        t1.start();
        t2.start();
        try {
            t1.join();
            t2.join();
        } catch (InterruptedException e) {
            System.err.println("Data load interrupted: " + e.getMessage());
            loadError.set(true);
        }
        if (loadError.get()) {
            System.exit(1);
        }

        // Launch UI
        ConsoleUI ui = new ConsoleUI(catalog, rentalService);
        ui.start();
    }
}
