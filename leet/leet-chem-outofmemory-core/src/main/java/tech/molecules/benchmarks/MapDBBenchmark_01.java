package tech.molecules.benchmarks;

import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.HTreeMap;
import org.mapdb.Serializer;

import java.util.*;

public class MapDBBenchmark_01 {

    public static void main(String[] args) {

        Random random = new Random(1234);

        // Number of objects to create and store in the benchmark
        int numberOfBulks          = 10;
        int numObjectsPerBulk      = 100000;
        int valueStringLength = 20;

        // Number of objects to fetch:
        int numberOfFetchedObjects = 1000000;

        // Create a new or open an existing database file
        DB db = DBMaker.fileDB("mydb.db").make();

        // Create or open a HashMap with String keys and String values
        HTreeMap<String, String> map = db.hashMap("myMap")
                .keySerializer(Serializer.STRING)
                .valueSerializer(Serializer.STRING)
                .createOrOpen();

        // Start benchmark for insertion
        long startInsertion = System.currentTimeMillis();

        // Insert objects into the map
        for (int bi = 0;bi<numberOfBulks;bi++) {
            Map<String,String> bda = generateBulkData(random,bi*numObjectsPerBulk,(bi+1)*numObjectsPerBulk,valueStringLength);
            map.putAll(bda);
        }

        long endInsertion = System.currentTimeMillis();
        long insertionTime = endInsertion - startInsertion;

        System.out.println("Insertion benchmark completed in " + insertionTime + " ms");


        // Retrieve objects from the map

        List<String> keysToFetch = new ArrayList<>();
        for (int i = 0; i < numberOfFetchedObjects; i++) {
            String key = "Key" + random.nextInt(numberOfBulks*numObjectsPerBulk);
            keysToFetch.add(key);
        }
        // Start benchmark for retrieval
        long startRetrieval = System.currentTimeMillis();
        for(int zi=0;zi<keysToFetch.size();zi++) {
            String value = map.get(keysToFetch.get(zi));
        }

        long endRetrieval = System.currentTimeMillis();
        long retrievalTime = endRetrieval - startRetrieval;

        System.out.println("Retrieval benchmark completed in " + retrievalTime + " ms");
        System.out.println(String.format("Retrieval: %f / ms" , (1.0*keysToFetch.size()) / retrievalTime ) );

        // Close the database
        db.close();
    }

    private static Map<String, String> generateBulkData(Random random, int start, int end, int length) {
        // Generate bulk data with pre-populated key-value pairs
        Map<String, String> bulkData = new HashMap<>(end-start);
        for (int i = start; i < end; i++) {
            String key = "Key" + i;
            String value = generateRandomString(random, length);
            bulkData.put(key, value);
        }
        return bulkData;
    }

    private static String generateRandomString(Random random, int length) {
        // Generate a random string of given length
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            char ch = (char) (random.nextInt(26) + 'a'); // Random lowercase letter
            sb.append(ch);
        }
        return sb.toString();
    }

}
