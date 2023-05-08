package tech.molecules.leet.chem.util;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class Parallelizer {

    public static <T> void computeParallelBlocking(Consumer<T> f, List<T> objects, int threads) throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        for (T obj : objects) {
            executor.submit(() -> f.accept(obj));
        }

        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    }

}
