package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.chem.ChemUtils;

import javax.swing.*;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

public class SimpleSARDecompositionModel {

    private int numThreads = 1;

    /**
     * Decompositions. The order defines the way in which the
     * unique series membership is defined (starting at first
     * decomposition)
     *
     */
    //private List<SimpleSARSeries> series;

    private SimpleSARProject project;


    private List<String> compounds = new ArrayList<>();

    private Map<String,Pair<SimpleSARSeries,SimpleSARDecomposition.SimpleSARResult>> firstMatches;
    private Map<SimpleSARSeries,List<SimpleSARDecomposition.SimpleSARResult>> seriesByFirstMatch;

    public SimpleSARDecompositionModel(List<SimpleSARSeries> series) {
        //this.series = series;
        this.project = new SimpleSARProject();
        this.project.setSeries(series);

        initThreadPool();
        initMonitorThread(500);
        firstMatches = new HashMap<>();
        seriesByFirstMatch = new HashMap<>();
    }

    // Create a ThreadPoolExecutor with n threads
    private ThreadPoolExecutor threadPool;

    private void initThreadPool() {
        int threads = numThreads;
        if(numThreads<=0) {threads = Runtime.getRuntime().availableProcessors();}
        threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads);
    }


    private Boolean newResults = false;

    /**
     * To not overload the system with tons of calls to the listener, we have a timed
     * check if there are new results
     *
     * @param milliseconds
     */
    private void initMonitorThread(int milliseconds) {
        Thread ti = new Thread() {
            @Override
            public void run() {
                while(true) {
                    try {
                        if(Thread.interrupted()) {return;}
                        Thread.sleep(milliseconds);

                        synchronized (newResults) {
                            if(newResults) {
                                fireNewDecompositions();
                            }
                            newResults = false;
                        }

                    } catch (InterruptedException e) {
                        //throw new RuntimeException(e);
                        e.printStackTrace();
                    }
                }
            }
        };
        ti.setPriority(Thread.MIN_PRIORITY);
        ti.start();
    }

    public List<SimpleSARSeries> getSeries() {
        return this.project.getSeries();
    }

    public List<SimpleSARDecomposition.SimpleSARResult> getSeriesDecomposition(SimpleSARSeries series) {
        List<SimpleSARDecomposition.SimpleSARResult> results = new ArrayList<>();
        synchronized (seriesByFirstMatch) {
            results = new ArrayList<>(this.seriesByFirstMatch.get(series));
        }
        return results;
    }

    public SimpleSARDecompositionModel getThisModel() {
        return this;
    }

    /**
     *
     *
     * @param newCompounds
     * @return future that returns when all computations for the given batch are done
     */
    public Future<Void> addCompounds(List<String> newCompounds) {
        CompletableFuture<Void> future = new CompletableFuture<>();

        int numTasks = newCompounds.size();
        AtomicInteger completedTasks = new AtomicInteger(0);

        for(String ci : newCompounds) {
            String fci = ci;
            List<SimpleSARSeries> fdecomps = new ArrayList<>(this.project.getSeries());
            Runnable ri = new Runnable() {
                @Override
                public void run() {
                    Pair<SimpleSARSeries,SimpleSARDecomposition.SimpleSARResult> ssi = computeFirstMatchingSARSeries(fdecomps,fci);
                    synchronized(firstMatches) {
                        firstMatches.put(fci, ssi);
                    }
                    synchronized(seriesByFirstMatch) {
                        seriesByFirstMatch.putIfAbsent(ssi.getLeft(),new ArrayList<>());
                        seriesByFirstMatch.get(ssi.getLeft()).add(ssi.getRight());
                    }
                    synchronized (newResults) {
                        newResults = true;
                    }
                }
            };
            this.threadPool.submit(() -> {
                try {
                    ri.run();
                } finally {
                    int completed = completedTasks.incrementAndGet();
                    if (completed == numTasks) {
                        future.complete(null); // All tasks completed
                    }
                }
            });
        }
        return future;
    }

    public static Pair<SimpleSARSeries,SimpleSARDecomposition.SimpleSARResult> computeFirstMatchingSARSeries(List<SimpleSARSeries> series, String mol) {
        StereoMolecule mi = ChemUtils.parseIDCode(mol);
        for(int zi=0;zi<series.size();zi++) {
            List<StereoMolecule> decomp_mols = series.get(zi).getSeriesDecomposition().getAllStructures();

            for(StereoMolecule di : decomp_mols) {
                StereoMolecule decomp = di;
                List<SimpleSARDecomposition.SimpleSARResult> matches = SimpleSARDecomposition.matchSimpleSAR(decomp, mi);


                if (matches.isEmpty()) {
                    continue;
                } else {
                    return Pair.of( series.get(zi) , matches.get(0) );
                }
            }
        }
        return Pair.of(null,null);
    }


    public static interface DecompositionModelListener {
        public void newDecompositions();
    }

    private List<DecompositionModelListener> listeners = new ArrayList<>();

    public void addListener(DecompositionModelListener li) {
        this.listeners.add(li);
    }

    public boolean removeListener(DecompositionModelListener li) {
        return this.listeners.remove(li);
    }

    public void fireNewDecompositions() {
        for(DecompositionModelListener li : this.listeners) {
            li.newDecompositions();
        }
    }


}
