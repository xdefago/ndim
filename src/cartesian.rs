use num_traits::*;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

///
/// Represent coordinates in an N-dimensional space.
///
///
/// # Examples
///
/// ```
/// use ndim::Cartesian;
///
/// let c = Cartesian::new([1, 2, 3]);
/// assert_eq!(c.coordinates, [1, 2, 3]);
/// assert_eq!(c.dim(), 3);
///
/// let c = Cartesian::new([1.5, 2.8, 3.7, 4.9]);
/// assert_eq!(c.coordinates, [1.5, 2.8, 3.7, 4.9]);
/// assert_eq!(c.dim(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Cartesian<T, const N: usize> {
    pub coordinates: [T; N],
}

impl<T, const N: usize> Eq for Cartesian<T, N> where T: Eq {}

impl<T, const N: usize> Default for Cartesian<T, N>
where
    T: Default + Copy,
{
    fn default() -> Self {
        Self {
            coordinates: [T::default(); N],
        }
    }
}

impl<T, const N: usize> Cartesian<T, N> {
    ///
    /// Create a new coordinate with the given components.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - The components of the coordinate as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndim::Cartesian;
    ///
    /// let c = Cartesian::new([1, 2, 3]);
    /// assert_eq!(c.coordinates, [1, 2, 3]);
    /// assert_eq!(c.dim(), 3);
    /// assert_eq!(c.coordinates.len(), c.dim());
    /// ```
    pub fn new(coordinates: [T; N]) -> Self {
        Self { coordinates }
    }

    ///
    /// Returns the number of dimensions of the space.
    ///
    pub fn dim(&self) -> usize {
        N
    }

    ///
    /// Returns the number of components in the coordinate.
    /// This is synonymous with [`Self::dim`].
    ///
    pub fn len(&self) -> usize {
        N
    }
}

impl<T: PrimInt, const N: usize> Cartesian<T, N> {
    ///
    /// Returns the Manhattan distance from a coordinate to another.
    ///
    /// The Manhattan distance is defined as the minimum number of steps
    /// from one location to the other when moving only along one dimension
    /// at a time. This is can also be seen as a generalization of the Hamming
    /// distance between the coordinates.
    ///
    /// It is calculated as the sum of the absolute differences between the
    /// coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndim::Cartesian;
    ///
    /// let c1 = Cartesian::new([1, 2, 3]);
    /// let c2 = Cartesian::new([0, 0, 0]);
    /// assert_eq!(c1.manhattan_distance(c2), 6);
    /// ```
    pub fn manhattan_distance(&self, rhs: Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            let min_value = self.coordinates[i].min(rhs.coordinates[i]);
            let max_value = self.coordinates[i].max(rhs.coordinates[i]);
            sum = sum + (max_value - min_value);
        }
        sum
    }

    ///
    /// Returns the Chebyshev distance from a coordinate to another.
    ///
    /// The Chebyshev distance is defined as the maximum difference between
    /// the coordinates.
    /// In 2D, it corresponds to the minimum number of steps that a king must
    /// take to move between the two squares on a chessboard.
    ///
    /// ```
    /// use ndim::Cartesian;
    ///
    /// let c1 = Cartesian::new([1, 2, 3]);
    /// let c2 = Cartesian::new([0, 0, 0]);
    /// assert_eq!(c1.chebyshev_distance(c2), 3);
    /// ```
    pub fn chebyshev_distance(&self, rhs: Self) -> T {
        let mut max = T::zero();
        for i in 0..N {
            let min_value = self.coordinates[i].min(rhs.coordinates[i]);
            let max_value = self.coordinates[i].max(rhs.coordinates[i]);
            let value = max_value - min_value;
            if value > max {
                max = value;
            }
        }
        max
    }

    ///
    /// Converts the integer coordinate to a corresponding coordinate
    /// with floating point.
    ///
    pub fn as_float<F: Float>(&self) -> Cartesian<F, N> {
        let mut coordinates = [F::zero(); N];
        for i in 0..N {
            coordinates[i] = F::from(self.coordinates[i]).unwrap();
        }
        Cartesian::new(coordinates)
    }
}

impl<T: Float, const N: usize> Cartesian<T, N> {
    ///
    /// Returns the Euclidean distance from a coordinate to another.
    ///
    /// The Euclidean distance is defined as the length of the straight
    /// segment from one coordinate to the other, in Euclidean space.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndim::Cartesian;
    ///
    /// let c1 = Cartesian::new([3., 4.]);
    /// let c2 = Cartesian::new([0., 0.]);
    /// assert_eq!(c1.euclidean_distance(c2), 5.);
    /// ```
    ///
    pub fn euclidean_distance(&self, rhs: Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + (self.coordinates[i] - rhs.coordinates[i]).powi(2);
        }
        sum.sqrt()
    }

    ///
    /// Returns the length of the coordinate.
    ///
    pub fn norm(&self) -> T {
        self.dot_product(*self).sqrt()
    }
}

impl<T, const N: usize> Cartesian<T, N>
where
    T: Zero + PartialEq,
{
    pub fn is_zero(&self) -> bool {
        for i in 0..N {
            if self.coordinates[i] != T::zero() {
                return false;
            }
        }
        true
    }
}

impl<T, const N: usize> Cartesian<T, N>
where
    T: Zero + Copy,
{
    ///
    /// Returns a zero (null) coordinate.
    ///
    pub fn zero() -> Self {
        // TODO: is there any way to get this into a const or a lazy_static??!
        Self {
            coordinates: [T::zero(); N],
        }
    }
}

impl<T, const N: usize> Cartesian<T, N>
where
    T: Float,
{
    ///
    /// Returns true if all of the coordinates are finite.
    ///
    pub fn is_finite(&self) -> bool {
        for i in 0..N {
            if !self.coordinates[i].is_finite() {
                return false;
            }
        }
        true
    }

    ///
    /// Returns true if any of the coordinates is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndim::Cartesian;
    ///
    /// let c = Cartesian::new([0., 1., f64::NAN]);
    /// assert!(c.is_nan());
    ///
    /// let c = Cartesian::new([0., 1., 3.]);
    /// assert!(! c.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        for i in 0..N {
            if self.coordinates[i].is_nan() {
                return true;
            }
        }
        false
    }
}

impl<T, const N: usize> Cartesian<T, N>
where
    T: Num + Copy,
{
    pub fn dot_product(&self, rhs: Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.coordinates[i] * rhs.coordinates[i];
        }
        sum
    }
}

impl<T> Cartesian<T, 3>
where
    T: Num + Copy,
{
    pub fn cross_product(&self, rhs: Self) -> Self {
        Self {
            coordinates: [
                self.y() * rhs.z() - self.z() * rhs.y(),
                self.z() * rhs.x() - self.x() * rhs.z(),
                self.x() * rhs.y() - self.y() * rhs.x(),
            ],
        }
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Cartesian<T, N> {
    fn as_ref(&self) -> &[T; N] {
        &self.coordinates
    }
}

impl<T, const N: usize> AsRef<[T]> for Cartesian<T, N> {
    fn as_ref(&self) -> &[T] {
        &self.coordinates
    }
}

impl<T, const N: usize> PartialOrd for Cartesian<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut smaller = 0;
        let mut larger = 0;
        let mut equal = 0;
        for i in 0..N {
            match self.coordinates[i].partial_cmp(&other.coordinates[i]) {
                Some(std::cmp::Ordering::Less) => smaller += 1,
                Some(std::cmp::Ordering::Equal) => equal += 1,
                Some(std::cmp::Ordering::Greater) => larger += 1,
                None => return None,
            }
        }
        match (smaller, equal, larger) {
            (0, _, 0) => Some(std::cmp::Ordering::Equal),
            (0, _, _) => Some(std::cmp::Ordering::Greater),
            (_, _, 0) => Some(std::cmp::Ordering::Less),
            (_, _, _) => None,
        }
    }
}

impl<T, const N: usize> Index<usize> for Cartesian<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Cartesian<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

impl<T, const N: usize> Add for Cartesian<T, N>
where
    T: Num + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coordinates = [T::zero(); N];
        for i in 0..N {
            coordinates[i] = self.coordinates[i] + rhs.coordinates[i];
        }
        Self { coordinates }
    }
}

impl<T, const N: usize> AddAssign for Cartesian<T, N>
where
    T: Num + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coordinates[i] = self.coordinates[i] + rhs.coordinates[i];
        }
    }
}

impl<T, const N: usize> Sub for Cartesian<T, N>
where
    T: Num + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut coordinates = [T::zero(); N];
        for i in 0..N {
            coordinates[i] = self.coordinates[i] - rhs.coordinates[i];
        }
        Self { coordinates }
    }
}

impl<T, const N: usize> SubAssign for Cartesian<T, N>
where
    T: Num + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coordinates[i] = self.coordinates[i] - rhs.coordinates[i];
        }
    }
}

impl<T, const N: usize> Mul<T> for Cartesian<T, N>
where
    T: Num + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut coordinates = [T::zero(); N];
        for i in 0..N {
            coordinates[i] = self.coordinates[i] * rhs;
        }
        Self { coordinates }
    }
}

impl<T, const N: usize> MulAssign<T> for Cartesian<T, N>
where
    T: Num + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.coordinates[i] = self.coordinates[i] * rhs;
        }
    }
}

impl<T, const N: usize> Div<T> for Cartesian<T, N>
where
    T: Num + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut coordinates = [T::zero(); N];
        for i in 0..N {
            coordinates[i] = self.coordinates[i] / rhs;
        }
        Self { coordinates }
    }
}

impl<T, const N: usize> DivAssign<T> for Cartesian<T, N>
where
    T: Num + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for i in 0..N {
            self.coordinates[i] = self.coordinates[i] / rhs;
        }
    }
}

impl<T, U, const N: usize> Neg for Cartesian<T, N>
where
    T: Num + Neg<Output = U> + Copy,
    U: Num + Copy,
{
    type Output = Cartesian<U, N>;

    fn neg(self) -> Self::Output {
        let mut coordinates = [U::zero(); N];
        for i in 0..N {
            coordinates[i] = -self.coordinates[i];
        }
        Self::Output { coordinates }
    }
}

impl<T: Copy> From<T> for Cartesian<T, 1> {
    fn from(value: T) -> Self {
        Self {
            coordinates: [value],
        }
    }
}

impl<T: Copy> From<Cartesian<T, 1>> for (T,) {
    fn from(value: Cartesian<T, 1>) -> Self {
        (value.x(),)
    }
}

impl<T: Copy, const N: usize> From<[T; N]> for Cartesian<T, N> {
    fn from(coordinates: [T; N]) -> Self {
        Self { coordinates }
    }
}

impl<T: Copy, const N: usize> From<Cartesian<T, N>> for [T; N] {
    fn from(cartesian: Cartesian<T, N>) -> Self {
        cartesian.coordinates
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "approx")] {
        impl<T, const N: usize> approx::AbsDiffEq for Cartesian<T, N>
        where
            T: approx::AbsDiffEq + PartialOrd,
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                for i in 0..N {
                    if !self.coordinates[i].abs_diff_eq(&other.coordinates[i], epsilon) {
                        return false;
                    }
                }
                true
            }
        }

        impl <T, const N: usize> approx::RelativeEq for Cartesian<T, N>
        where
            T: approx::RelativeEq + PartialOrd,
            T::Epsilon: Copy
        {
            fn default_max_relative() -> Self::Epsilon {
                T::default_max_relative()
            }

            fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                for i in 0..N {
                    if !self.coordinates[i].relative_eq(&other.coordinates[i], epsilon, max_relative) {
                        return false;
                    }
                }
                true
            }
        }

        impl <T, const N: usize> approx::UlpsEq for Cartesian<T, N>
        where
            T: approx::UlpsEq + PartialOrd,
            T::Epsilon: Copy
        {
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                for i in 0..N {
                    if !self.coordinates[i].ulps_eq(&other.coordinates[i], epsilon, max_ulps) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "experimental")] {
        pub struct Assert<const COND: bool> {}
        pub trait IsTrue {}
        impl IsTrue for Assert<true> {}

        impl <T, const N: usize> Cartesian<T, N>
        where
            T: Copy,
            Assert::<{N >= 1}>: IsTrue
        {
            /// Returns the x coordinate.
            ///
            /// Not defined if N < 1.
            ///
            #[inline]
            fn x(&self) -> T {
                self.coordinates[0]
            }
        }

        impl <T, const N: usize> Cartesian<T, N>
        where
            T: Copy,
            Assert::<{N >= 2}>: IsTrue
        {
            /// Returns the y coordinate.
            ///
            /// Not defined if N < 2.
            ///
            #[inline]
            fn y(&self) -> T {
                self.coordinates[1]
            }
        }

        impl <T, const N: usize> Cartesian<T, N>
        where
            T: Copy,
            Assert::<{N >= 3}>: IsTrue
        {
            /// Returns the z coordinate.
            ///
            /// Not defined if N < 3.
            ///
            #[inline]
            fn z(&self) -> T {
                self.coordinates[2]
            }
        }
    } else {
        impl <T, const N: usize> Cartesian<T, N>
        where T: Copy {
            /// Returns the x coordinate.
            ///
            /// # Panics
            ///
            /// Panics if N < 1.
            ///
            pub fn x(&self) -> T {
                self.coordinates[0]
            }

            /// Returns the y coordinate.
            ///
            /// # Panics
            ///
            /// Panics if N < 2.
            ///
            pub fn y(&self) -> T {
                self.coordinates[1]
            }

            /// Returns the z coordinate.
            ///
            /// # Panics
            ///
            /// Panics if N < 3.
            ///
            pub fn z(&self) -> T {
                self.coordinates[2]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian() {
        let c = Cartesian::new([0.0, 1.0, 2.0]);
        assert_eq!(c.coordinates, [0.0, 1.0, 2.0]);

        let c = Cartesian::new([1, 2, 3]);
        assert_eq!(c.coordinates, [1, 2, 3]);
    }

    #[test]
    fn test_dot_product() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert_eq!(c1.dot_product(c2), 32);
    }

    #[test]
    fn test_default() {
        let c = Cartesian::<i32, 3>::default();
        assert_eq!(c.coordinates, [0, 0, 0]);

        let c = Cartesian::<f32, 4>::default();
        assert_eq!(c.coordinates, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert_eq!(c1 + c2, Cartesian::new([5, 7, 9]));
    }

    #[test]
    fn test_add_assign() {
        let mut c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        c1 += c2;
        assert_eq!(c1, Cartesian::new([5, 7, 9]));
    }

    #[test]
    fn test_sub() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert_eq!(c1 - c2, Cartesian::new([-3, -3, -3]));
    }

    #[test]
    fn test_sub_assign() {
        let mut c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        c1 -= c2;
        assert_eq!(c1, Cartesian::new([-3, -3, -3]));
    }

    #[test]
    fn test_mul() {
        let c = Cartesian::new([1, 2, 3]);
        assert_eq!(c * 2, Cartesian::new([2, 4, 6]));
    }

    #[test]
    fn test_mul_assign() {
        let mut c = Cartesian::new([1, 2, 3]);
        c *= 2;
        assert_eq!(c, Cartesian::new([2, 4, 6]));
    }

    #[test]
    fn test_div() {
        let c = Cartesian::new([1, 2, 3]);
        assert_eq!(c / 2, Cartesian::new([0, 1, 1]));
    }

    #[test]
    fn test_div_assign() {
        let mut c = Cartesian::new([1, 2, 3]);
        c /= 2;
        assert_eq!(c, Cartesian::new([0, 1, 1]));
    }

    #[test]
    fn test_index() {
        let c = Cartesian::new([1, 2, 3]);
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 2);
        assert_eq!(c[2], 3);
    }

    #[test]
    fn test_index_mut() {
        let mut c = Cartesian::new([1, 2, 3]);
        c[0] = 4;
        c[1] = 5;
        c[2] = 6;
        assert_eq!(c, Cartesian::new([4, 5, 6]));
    }

    #[test]
    fn test_manhattan_distance() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert_eq!(c1.manhattan_distance(c2), 9);
    }

    #[test]
    fn test_euclidean_distance() {
        let c1 = Cartesian::new([1., 2., 3.]);
        let c2 = Cartesian::new([4., 5., 6.]);
        approx::assert_abs_diff_eq!(c1.euclidean_distance(c2), 5.196152422706632);
    }

    #[test]
    fn test_chebyshev_distance() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert_eq!(c1.chebyshev_distance(c2), 3);
    }

    #[test]
    fn test_coord_xyz() {
        let c1 = Cartesian::new([1]);
        let c2 = Cartesian::new([1, 2]);
        let c3 = Cartesian::new([1, 2, 3]);
        let c4 = Cartesian::new([1, 2, 3, 4]);

        assert_eq!(c1.x(), 1);
        assert_eq!(c2.x(), 1);
        assert_eq!(c3.x(), 1);
        assert_eq!(c4.x(), 1);

        assert_eq!(c2.y(), 2);
        assert_eq!(c3.y(), 2);
        assert_eq!(c4.y(), 2);

        assert_eq!(c3.z(), 3);
        assert_eq!(c4.z(), 3);
    }

    #[test]
    fn test_zero() {
        let zero = Cartesian::zero();
        assert_eq!(zero, Cartesian::new([0, 0, 0]));
        assert!(zero.is_zero());

        let zero = Cartesian::zero();
        assert_eq!(zero, Cartesian::new([0., 0., 0.]));
        assert!(zero.is_zero());
        assert!(zero.is_finite());
        assert!(!zero.is_nan());
    }

    #[test]
    fn test_nan() {
        let c = Cartesian::new([1.0, 2.0, 3.0]);
        assert!(c.is_finite());
        assert!(!c.is_nan());

        let c = Cartesian::new([1.0, 2.0, std::f64::NAN]);
        assert!(!c.is_finite());
        assert!(c.is_nan());

        let c = Cartesian::new([1.0, 2.0, std::f32::NAN]);
        assert!(!c.is_finite());
        assert!(c.is_nan());

        let c = Cartesian::new([1.0, 2.0, std::f64::INFINITY]);
        assert!(!c.is_finite());
        assert!(!c.is_nan());

        let c = Cartesian::new([1.0, 2.0, std::f32::INFINITY]);
        assert!(!c.is_finite());
        assert!(!c.is_nan());
    }

    #[test]
    fn test_partial_ord() {
        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([4, 5, 6]);
        assert!(c1 < c2);
        assert!(c2 > c1);
        assert!(c1 <= c2);
        assert!(c2 >= c1);

        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([1, 2, 3]);
        assert!(c1 == c2);
        assert!(c1 <= c2);
        assert!(c1 >= c2);
        assert!(!(c1 < c2));
        assert!(!(c1 > c2));

        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([1, 2, 4]);
        assert!(c1 < c2);
        assert!(c1 <= c2);
        assert!(c1 != c2);

        let c1 = Cartesian::new([1, 2, 3]);
        let c2 = Cartesian::new([3, 2, 2]);
        assert!(!(c1 > c2));
        assert!(!(c1 < c2));
        assert!(c1 != c2);
    }

    #[cfg(feature = "approx")]
    #[test]
    fn test_approx_eq() {
        use approx::*;

        let c1 = Cartesian::new([1.0, 2.0, 3.0]);
        let c2 = Cartesian::new([1.0, 2.0, 3.0000000000000001]);
        let c3 = Cartesian::new([1.0, 2.0, 3.000001]);
        assert_abs_diff_eq!(c1, c2);
        assert_abs_diff_ne!(c1, c3);
        assert_relative_eq!(c1, c2);
        assert_relative_ne!(c1, c3);
        assert_ulps_eq!(c1, c2);
        assert_ulps_ne!(c1, c3);

        assert_abs_diff_eq!(c1, c3, epsilon = 0.00001);
        assert_relative_eq!(c1, c3, epsilon = 0.00001);
        assert_ulps_eq!(c1, c3, epsilon = 0.00001);

        assert_abs_diff_ne!(c1, c3, epsilon = 0.000001);
        assert_relative_ne!(c1, c3, epsilon = 0.000001);
        assert_ulps_ne!(c1, c3, epsilon = 0.000001);
    }

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Cartesian<f64, 1>>();
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Cartesian<f64, 1>>();
    }
}
