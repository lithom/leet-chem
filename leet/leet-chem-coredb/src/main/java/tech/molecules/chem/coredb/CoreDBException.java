package tech.molecules.chem.coredb;

public class CoreDBException extends Exception {
    public CoreDBException() {
    }

    public CoreDBException(String message) {
        super(message);
    }

    public CoreDBException(String message, Throwable cause) {
        super(message, cause);
    }

    public CoreDBException(Throwable cause) {
        super(cause);
    }

    public CoreDBException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
