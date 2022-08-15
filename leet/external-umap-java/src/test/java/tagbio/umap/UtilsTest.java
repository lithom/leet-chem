package tagbio.umap;

import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;

import junit.framework.TestCase;

public class UtilsTest extends TestCase {

  public void testRejectionSample() {
    final Random r = new Random();
    final int[] rs = Utils.rejectionSample(20, 100, r);
    assertEquals(20, rs.length);
    final TreeSet<Integer> uniq = new TreeSet<>();
    for (final int v : rs) {
      assertTrue(v >= 0 && v < 100);
      uniq.add(v);
    }
    assertEquals(20, uniq.size());
    try {
      Utils.rejectionSample(5, 2, r);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testSplitRandom() {
    Random[] randoms = Utils.splitRandom(new Random(543), 5);
    assertEquals(5, randoms.length);
    final int[] expected = new int[]{-1797116241, -80536573, -1257863196, 1902860816, 160052042, -993477666, -1141936413, 1152672626, -749860475, -1591028618};
    for (int i = 0; i < randoms.length; i++) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
    randoms = Utils.splitRandom(new Random(543), 7);
    assertEquals(7, randoms.length);
    for (int i = randoms.length - 1; i >= 0; i--) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
    randoms = Utils.splitRandom(new Random(543), 10);
    assertEquals(10, randoms.length);
    for (int i = 0; i < randoms.length; i++) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
        randoms = Utils.splitRandom(new Random(345), 10);
    assertEquals(10, randoms.length);
    for (int i = 0; i < randoms.length; i++) {
      assertNotSame(expected[i], randoms[i].nextInt());
    }
  }

  public void testNow() {
    assertTrue(Utils.now().matches("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9] "));
  }

  public void testFastKnnIndices() {
    final Matrix m = new DefaultMatrix(new float[][] {{1, 2}, {0, 1}, {1, 0}, {0, 0}, {2, 1}});
    final int[][] knn = Utils.fastKnnIndices(m, 2);
    assertEquals("[[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]", Arrays.deepToString(knn));
  }

  public void testL2Norm() {
    final float[] vec = new float[2];
    assertEquals(0.0, Utils.norm(vec), 1e-6);
    vec[0] = 1;
    assertEquals(1.0, Utils.norm(vec), 1e-6);
    vec[1] = 1;
    assertEquals(Math.sqrt(2), Utils.norm(vec), 1e-6);
    vec[0] = 3;
    vec[1] = 4;
    assertEquals(5.0, Utils.norm(vec), 1e-6);
  }
}

