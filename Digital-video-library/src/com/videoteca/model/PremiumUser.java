package com.videoteca.model;

/**
 * Premium users may have up to 5 active rentals.
 */
public class PremiumUser extends User {
    private static final int MAX_ACTIVE = 5;

    public PremiumUser(String userId, String name) {
        super(userId, name);
    }

    public PremiumUser(String name) {
        super(name);
    }

    @Override
    public boolean canRentMore(int activeRentals) {
        return activeRentals < MAX_ACTIVE;
    }
}
