// src/com/videoteca/config/AppConfig.java
package com.videoteca.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Loads application configuration from config.properties in resources.
 */
public class AppConfig {
    private static final Properties props = new Properties();

    static {
        try (InputStream in = AppConfig.class.getClassLoader()
                                       .getResourceAsStream("config.properties")) {
            if (in == null) {
                throw new RuntimeException("Missing config.properties in classpath");
            }
            props.load(in);
        } catch (IOException e) {
            throw new ExceptionInInitializerError("Failed to load config.properties: " + e);
        }
    }

    /**
     * Returns the property value for the given key, or null if not found.
     */
    public static String get(String key) {
        return props.getProperty(key);
    }

    /**
     * Returns the integer property for the given key, or the default value if not found or invalid.
     */
    public static int getInt(String key, int defaultValue) {
        String v = props.getProperty(key);
        try {
            return (v != null) ? Integer.parseInt(v) : defaultValue;
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}