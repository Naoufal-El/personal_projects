package com.videoteca.model;

public class AdminUser extends User {
    /**
     * Constructor used when loading from DB (with assigned userId).
     */
    public AdminUser(String userId, String name) {
        super(userId, name);
    }

    /**
     * Constructor for new admin before DB assignment.
     */
    public AdminUser(String name) {
        super(name);
    }

    @Override
    public boolean canRentMore(int currentActiveRentals) {
        return true; // unlimited rentals
    }

    /**
     * Admin-specific capability check for managing all resources.
     */
    public boolean canManageAll() {
        return true;
    }
}
