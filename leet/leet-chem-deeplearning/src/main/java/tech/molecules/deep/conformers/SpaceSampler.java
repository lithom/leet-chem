package tech.molecules.deep.conformers;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;

import java.util.*;

public class SpaceSampler {

    private StereoMolecule mi;

    private int maxAtoms;

    private double cubeLengthAnstrom;
    private int    gridSize;

    private double varianceA;
    private double varianceB;
    private double offsetA;


    /**
     *
     *
     * @param mi
     * @param maxAtoms
     * @param cubeLengthAnstrom
     * @param gridSize
     * @param varianceA
     * @param varianceB
     * @param offsetA is relative in varianceA, i.e. 0.25 means centers are shifted randomly by  rand( 0.25*variance ) in each dimension
     */
    public SpaceSampler(StereoMolecule mi, int maxAtoms, double cubeLengthAnstrom, int gridSize, double varianceA, double varianceB, double offsetA) {
        this.mi = mi;
        this.maxAtoms = maxAtoms;
        this.cubeLengthAnstrom = cubeLengthAnstrom;
        this.gridSize = gridSize;
        this.varianceA = varianceA;
        this.varianceB = varianceB;

        this.mi.center();

        // apply random transformation?
        if(true) {
            Random ri = new Random();
            double rotX = ri.nextDouble(); double rotY = ri.nextDouble(); double rotZ = ri.nextDouble();
            for(int zi=0;zi<mi.getAllAtoms();zi++) {
                double xx = mi.getAtomX(zi); double yy = mi.getAtomY(zi); double zz = mi.getAtomZ(zi);
                double xr[] = rotate3D(xx,yy,zz,rotX,rotY,rotZ);
                this.mi.setAtomX(zi,xr[0]); this.mi.setAtomY(zi,xr[1]); this.mi.setAtomZ(zi,xr[2]);
            }
        }

        initAtomEncoding();
        initBondsEncoding();
    }

    public static class SampledSpace {
        public final double[][][][]   x;
        public final int[]            structureInfo; // element
        public final int[][]          bondsType; //inbetween atom x and y
        public SampledSpace(double[][][][] x, int[] structureInfo, int[][] bondsType) {
            this.x = x;
            this.structureInfo = structureInfo;
            this.bondsType = bondsType;
        }
    }

    public static class SampledSpaceOneHot {
        public final double[][][][]   x;
        public final double[][][][]   x_target;
        public final boolean[][]      structureInfo; // element
        public final boolean[][][]    bondsType; // inbetween atom x and y
        public SampledSpaceOneHot(double[][][][] x, double[][][][] x_target, int[] structureInfo, int[][] bondsType) {
            this.x = x;
            this.x_target = x_target;
            this.structureInfo = new boolean[structureInfo.length][10];
            this.bondsType     = new boolean[bondsType.length][bondsType.length][7];
            for(int zi=0;zi<structureInfo.length;zi++) {
                if(structureInfo[zi]!=0) { this.structureInfo[zi][structureInfo[zi]] = true;}
            }
            for(int zi=0;zi<bondsType.length;zi++) {
                for(int zj=0;zj<bondsType[0].length;zj++) {
                    if (bondsType[zi][zj] != 0) {
                        this.bondsType[zi][zj][bondsType[zi][zj]] = true;
                    }
                }
            }
        }
    }

    // use int for one-hot encoding because hdf5 only knows int and not boolean.
    // But the arrays contain just boolean information.
    public static class SampledSpaceOneHot2 {
        public final double[][][][]   x;
        public final double[][][][]   x_target;
        public final int[][]      structureInfo; // element
        public final int[][][]    bondsType; // inbetween atom x and y
        public SampledSpaceOneHot2(double[][][][] x, double[][][][] x_target, int[] structureInfo, int[][] bondsType) {
            this.x = x;
            this.x_target = x_target;
            this.structureInfo = new int[structureInfo.length][9];
            this.bondsType     = new int[bondsType.length][bondsType.length][7];
            for(int zi=0;zi<structureInfo.length;zi++) {
                if(structureInfo[zi]!=0) { this.structureInfo[zi][structureInfo[zi]] = 1;}
            }
            for(int zi=0;zi<bondsType.length;zi++) {
                for(int zj=0;zj<bondsType[0].length;zj++) {
                    if (bondsType[zi][zj] != 0) {
                        this.bondsType[zi][zj][bondsType[zi][zj]] = 1;
                    }
                }
            }
        }
    }


    private Map<Integer,Integer> atomEncoding;
    private Map<Integer,Integer> bondEncoding;

    private void initAtomEncoding() {
        atomEncoding = new HashMap<>();
        atomEncoding.put(1,1);
        atomEncoding.put(6,2);
        atomEncoding.put(7,3);
        atomEncoding.put(8,4);
        atomEncoding.put(9,5);
        atomEncoding.put(15,6);
        atomEncoding.put(16,7);
        atomEncoding.put(17,8);
        atomEncoding.put(35,9);
    }

    private void initBondsEncoding() {
        bondEncoding = new HashMap<>();
        bondEncoding.put(Molecule.cBondTypeSingle,1);
        bondEncoding.put(Molecule.cBondTypeDelocalized,2);
        bondEncoding.put(Molecule.cBondTypeDouble,3);
        bondEncoding.put(Molecule.cBondTypeTriple,4);
        bondEncoding.put(Molecule.cBondTypeUp,5);
        bondEncoding.put(Molecule.cBondTypeDown,6);
    }

    private int encodeAtom(int atomicNo) throws Exception {
        Integer xi = atomEncoding.get(atomicNo);
        if(xi==null) {throw new Exception("Unknown element: "+atomicNo);}
        return xi;
    }
    private int encodeBond(int bondType) throws Exception {
        Integer xi = bondEncoding.get(bondType);
        if(xi==null) {throw new Exception("Unknown bond type: "+bondType);}
        return xi;
    }

    /**
     * [0]: before, [1]: after
     *
     * Random ri is for shift of A.
     *
     * @return
     * @throws Exception
     */
    public SampledSpaceOneHot sampleSpace(Random ri) throws Exception {
        this.mi.ensureHelperArrays(Molecule.cHelperCIP);

        double[][][][]    xA = new double[gridSize][gridSize][gridSize][maxAtoms];
        double[][][][]    xB = new double[gridSize][gridSize][gridSize][maxAtoms];
        int[]    atoms       = encodeAtoms();
        int[][]  bonds       = encodeBonds();

        // now sample the space:




        // 1. find bounding box:
        double xmin=Double.POSITIVE_INFINITY; double xmax=Double.NEGATIVE_INFINITY;
        double ymin=Double.POSITIVE_INFINITY; double ymax=Double.NEGATIVE_INFINITY;
        double zmin=Double.POSITIVE_INFINITY; double zmax=Double.NEGATIVE_INFINITY;

        for(int zi=0;zi<mi.getAtoms();zi++) {
            xmin = Math.min(xmin,mi.getAtomX(zi));xmax = Math.max(xmax,mi.getAtomZ(zi));
            ymin = Math.min(ymin,mi.getAtomY(zi));ymax = Math.max(ymax,mi.getAtomY(zi));
            zmin = Math.min(zmin,mi.getAtomZ(zi));zmax = Math.max(zmax,mi.getAtomZ(zi));
        }

        // check if molecule fits..
        List<Double> boundValues = new ArrayList<>();
        boundValues.add(xmin);boundValues.add(xmax);
        boundValues.add(ymin);boundValues.add(ymax);
        boundValues.add(zmin);boundValues.add(zmax);
        boolean toobig = boundValues.stream().mapToDouble(xi -> Math.abs(xi)).anyMatch( xi -> xi > gridSize );
        if(toobig) {
            throw new Exception("Too big: "+boundValues.stream().mapToDouble(xi -> Math.abs(xi)).max().getAsDouble());
        }


        double x0 = -0.5*cubeLengthAnstrom;
        double y0 = -0.5*cubeLengthAnstrom;
        double z0 = -0.5*cubeLengthAnstrom;

        double grid = cubeLengthAnstrom / gridSize;

        for(int ai=0;ai<mi.getAllAtoms();ai++) {
            double volTotA = 0.0;
            double volTotB = 0.0;
            double px = mi.getAtomX(ai);
            double py = mi.getAtomY(ai);
            double pz = mi.getAtomZ(ai);

            for(int xi=0;xi<gridSize;xi++) {
                for(int yi=0;yi<gridSize;yi++) {
                    for(int zi=0;zi<gridSize;zi++) {
                        double xx = x0 + xi*grid;
                        double yy = y0 + yi*grid;
                        double zz = z0 + zi*grid;

                        double xxA = xx + ri.nextDouble() * this.varianceA * this.offsetA;
                        double yyA = yy + ri.nextDouble() * this.varianceA * this.offsetA;
                        double zzA = zz + ri.nextDouble() * this.varianceA * this.offsetA;

                        double dxA = (px-xxA); double dyA = (py-yyA); double dzA = (pz-zzA);
                        double dx  = (px-xx);  double dy = (py-yy);   double dz = (pz-zz);
                        double distA = Math.sqrt( dxA*dxA+dyA*dyA+dzA*dzA );
                        double distB = Math.sqrt( dx*dx+dy*dy+dz*dz );

                        if(distA < varianceA) {
                            xA[xi][yi][zi][ai] = 1.0;
                            volTotA += 1.0;
                        }
                        else {
                            double vi = Math.max( 0 , 1.0 - (distA-varianceA) );
                            xA[xi][yi][zi][ai] = vi;
                            volTotA += vi;
                        }
                        if(distB < varianceB) {
                            xB[xi][yi][zi][ai] = 1.0;
                            volTotB += 1.0;
                        }
                        else {
                            double vi = Math.max( 0 , 1.0 - (distA-varianceB) );
                            xA[xi][yi][zi][ai] = vi;
                            volTotA += vi;
                        }
                    }
                }
            }
            // Now: normalize? Maybe not..
            int abc = 3;
        }

        //return new SampledSpaceOneHot[]{ new SampledSpaceOneHot(xA,atoms,bonds) , new SampledSpaceOneHot(xB,atoms,bonds) };
        return new SampledSpaceOneHot(xA,xB,atoms,bonds);
    }


    private int[] encodeAtoms() throws Exception {

        int[] atoms = new int[maxAtoms];
        for(int zi=0;zi<mi.getAllAtoms();zi++) {
            atoms[zi] = encodeAtom( mi.getAtomicNo(zi) );
        }
        return atoms;
    }

    private int[][] encodeBonds() throws Exception {
        int[][] bonds = new int[maxAtoms][maxAtoms];
        for(int zi=0;zi<mi.getAllBonds();zi++) {
            int xa = mi.getBondAtom(0,zi);
            int xb = mi.getBondAtom(1,zi);
            int bondType = mi.getBondType(zi);
            int encoded = encodeBond(bondType);
            bonds[xa][xb] = encoded;
            bonds[xb][xa] = encoded;
            //for(int zj=zi+1;zj<mi.getAtoms();zj++) {
            //    bonds[zi][zj] = mi.getBondType(mi.getBond(zi,zj));
            //}
        }
        return bonds;
    }

    public static double[] rotate3D(double x, double y, double z, double rotationX, double rotationY, double rotationZ) {
        // Convert rotation angles from degrees to radians
        double radRotationX = Math.toRadians(rotationX);
        double radRotationY = Math.toRadians(rotationY);
        double radRotationZ = Math.toRadians(rotationZ);

        // Apply 3D rotation formulas
        double rotatedX = x * Math.cos(radRotationY) * Math.cos(radRotationZ)
                - y * (Math.cos(radRotationX) * Math.sin(radRotationZ)
                - Math.sin(radRotationX) * Math.sin(radRotationY) * Math.cos(radRotationZ))
                + z * (Math.sin(radRotationX) * Math.sin(radRotationZ)
                + Math.cos(radRotationX) * Math.sin(radRotationY) * Math.cos(radRotationZ));

        double rotatedY = x * Math.cos(radRotationY) * Math.sin(radRotationZ)
                + y * (Math.cos(radRotationX) * Math.cos(radRotationZ)
                + Math.sin(radRotationX) * Math.sin(radRotationY) * Math.sin(radRotationZ))
                - z * (Math.sin(radRotationX) * Math.cos(radRotationZ)
                - Math.cos(radRotationX) * Math.sin(radRotationY) * Math.sin(radRotationZ));

        double rotatedZ = -x * Math.sin(radRotationY)
                + y * Math.sin(radRotationX) * Math.cos(radRotationY)
                + z * Math.cos(radRotationX) * Math.cos(radRotationY);

        return new double[]{rotatedX, rotatedY, rotatedZ};
    }

}
