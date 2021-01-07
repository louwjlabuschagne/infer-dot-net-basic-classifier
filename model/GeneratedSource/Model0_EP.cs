// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace Models
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 0.3.1912.403 at 13:24 on donderdag 7 januari 2021.
	/// </remarks>
	public partial class Model0_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>Field backing the bothHeads property</summary>
		private bool BothHeads;
		public Bernoulli bothHeads_marginal;
		/// <summary>True if Changed_bothHeads has executed. Set this to false to force re-execution of Changed_bothHeads</summary>
		public bool Changed_bothHeads_isDone;
		/// <summary>Message to marginal of 'firstCoin'</summary>
		public Bernoulli firstCoin_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Message to marginal of 'secondCoin'</summary>
		public Bernoulli secondCoin_marginal_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'bothHeads'</summary>
		public bool bothHeads
		{
			get {
				return this.BothHeads;
			}
			set {
				if (this.BothHeads!=value) {
					this.BothHeads = value;
					this.numberOfIterationsDone = 0;
					this.Changed_bothHeads_isDone = false;
				}
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		#endregion

		#region Methods
		/// <summary>
		/// Returns the marginal distribution for 'bothHeads' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli BothHeadsMarginal()
		{
			return this.bothHeads_marginal;
		}

		/// <summary>Computations that depend on the observed value of bothHeads</summary>
		private void Changed_bothHeads()
		{
			if (this.Changed_bothHeads_isDone) {
				return ;
			}
			this.bothHeads_marginal = Bernoulli.Uniform();
			this.bothHeads_marginal = Distribution.SetPoint<Bernoulli,bool>(this.bothHeads_marginal, this.BothHeads);
			Bernoulli vBernoulli0 = Bernoulli.Uniform();
			this.firstCoin_marginal_F = Bernoulli.Uniform();
			Bernoulli firstCoin_use_B = default(Bernoulli);
			// Message to 'firstCoin_use' from And factor
			firstCoin_use_B = BooleanAndOp.AAverageConditional(this.BothHeads, vBernoulli0);
			// Message to 'firstCoin_marginal' from Variable factor
			this.firstCoin_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(firstCoin_use_B, vBernoulli0, this.firstCoin_marginal_F);
			this.secondCoin_marginal_F = Bernoulli.Uniform();
			Bernoulli secondCoin_use_B = default(Bernoulli);
			// Message to 'secondCoin_use' from And factor
			secondCoin_use_B = BooleanAndOp.BAverageConditional(this.BothHeads, vBernoulli0);
			// Message to 'secondCoin_marginal' from Variable factor
			this.secondCoin_marginal_F = VariableOp.MarginalAverageConditional<Bernoulli>(secondCoin_use_B, vBernoulli0, this.secondCoin_marginal_F);
			this.Changed_bothHeads_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_bothHeads();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>
		/// Returns the marginal distribution for 'firstCoin' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli FirstCoinMarginal()
		{
			return this.firstCoin_marginal_F;
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="bothHeads") {
				return this.bothHeads;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="bothHeads") {
				return this.BothHeadsMarginal();
			}
			if (variableName=="firstCoin") {
				return this.FirstCoinMarginal();
			}
			if (variableName=="secondCoin") {
				return this.SecondCoinMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>
		/// Returns the marginal distribution for 'secondCoin' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public Bernoulli SecondCoinMarginal()
		{
			return this.secondCoin_marginal_F;
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="bothHeads") {
				this.bothHeads = (bool)value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}
