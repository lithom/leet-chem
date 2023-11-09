package tech.molecules.leet.chem.util;

import com.actelion.research.chem.SSSearcher;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Semaphore;

public class PoolManager<T> {

    private final List<T> instances;
    private final Semaphore available;
    private boolean[] used;

    public PoolManager(List<T> instances) {
        this.instances = instances;
        this.available = new Semaphore(this.instances.size(), true);
        this.used = new boolean[instances.size()];
    }

    public T acquireSearcher() throws InterruptedException {
        available.acquire();
        synchronized(this) {
            return getNextAvailableInstance();
        }
    }

    public void releaseSearcher(SSSearcher searcher) {
        synchronized (this) {
            if (markAsUnused(searcher))
                available.release();
        }
    }

    private T getNextAvailableInstance() {
        for (int i = 0; i < this.instances.size(); ++i) {
            if (!used[i]) {
                used[i] = true;
                return this.instances.get(i);
            }
        }
        return null; // Not reached
    }

    private boolean markAsUnused(SSSearcher searcher) {
        for (int i = 0; i < this.instances.size(); ++i) {
            if (this.instances.get(i) == searcher) {
                if (used[i]) {
                    used[i] = false;
                    return true;
                } else
                    return false;
            }
        }
        return false;
    }

}
