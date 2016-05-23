    ''' <summary>
    ''' Create and train an neural network 
    ''' </summary>
    ''' <remarks></remarks>
    Private Class ClassNeuralNetwork
        Public Structure NeuralNetwork
            ''' <summary>
            ''' Creates a Layer of the Neural Network 
            ''' </summary>
            ''' <param name="nLayerType">Input / Hidden / Output</param>
            ''' <param name="NumberOfNodes">Amount of Nodes in the layer</param>
            ''' <param name="ActivationFunct">Activation FunctionType</param>
            ''' <returns>Created Layer</returns>
            ''' <remarks>there are only three layers in this network</remarks>
            Private Shared Function CreateLayer(ByRef nLayerType As LayerType,
                                        ByRef NumberOfNodes As Integer,
                               ActivationFunct As TransferFunctionType) As Layer

                Dim nLayer As New Layer
                Dim Nodes As New List(Of Neuron)
                Dim Node As New Neuron

                'Create nodes
                For i = 1 To NumberOfNodes
                    Nodes.Add(Node)
                Next
                nLayer.ActivationFunction = ActivationFunct
                nLayer.Nodes = Nodes
                nLayer.NumberOfNodes = NumberOfNodes
                nLayer.LayerType = nLayerType
                Return nLayer
            End Function
            ''' <summary>
            ''' creates neural network
            ''' can create a deep belief network or simple three layered network
            ''' </summary>
            ''' <param name="NetworkParameters">Parameters required to create the network</param>
            ''' <returns>created network</returns>
            ''' <remarks>requires training
            ''' generated neural network, number of nodes in hidden layers is calculated by this generation algorithm</remarks>
            Public Shared Function CreateNeuralNetwork(ByRef NetworkParameters As NeuralNetworkParameters) As NeuralNetwork
                Dim NN As New NeuralNetwork
                Dim HiddenLayer As New Layer
                'Number of hidden nodes is calculated by the mean of the input and output nodes
                Dim NumberOfHiddenNodes As Integer = NeuralNetwork.CalculateNumberOfHiddenNodes(NetworkParameters.NumberOfInputs, NetworkParameters.NumberOfOutputs)


                'Create InputLayer nodes
                NN.InputLayer = CreateLayer(LayerType.Input, NetworkParameters.NumberOfInputs, TransferFunctionType.none)
                For Each node As Neuron In NN.InputLayer.Nodes
                    node.weight = CreateRandWeight(NetworkParameters.NumberOfInputs, NumberOfHiddenNodes)
                Next


                'Create hidden layers
                For i = 1 To NetworkParameters.NumberOfHiddenLayers
                    'Create HiddenLayer nodes
                    HiddenLayer = CreateLayer(LayerType.Hidden, NumberOfHiddenNodes, NetworkParameters.HiddenLayerFunctionType)
                    For Each node As Neuron In HiddenLayer.Nodes
                        node.weight = CreateRandWeight(NetworkParameters.NumberOfInputs, NumberOfHiddenNodes)
                    Next
                    NN.HiddenLayers.Add(HiddenLayer)
                Next


                'Create OutputLayer nodes
                NN.OutputLayer = CreateLayer(LayerType.Output, NetworkParameters.NumberOfOutputs, NetworkParameters.OutputLayerFunctionType)



                'Return DeepBelief Network(neuralNetwork)
                Return NN
            End Function

            ''' <summary>
            ''' These are the parameters of the network to be created
            ''' </summary>
            ''' <remarks></remarks>
            Public Structure NeuralNetworkParameters
                'Network Parameters
                Public NumberOfInputs As Integer
                Public NumberOfOutputs As Integer
                Public NumberOfHiddenLayers As Integer
                Public OutputLayerFunctionType As TransferFunctionType
                Public HiddenLayerFunctionType As TransferFunctionType
            End Structure
            'Data shapes and structures
            ''' <summary>
            ''' Neural network Layer types
            ''' </summary>
            ''' <remarks></remarks>
            Enum LayerType
                Input
                Hidden
                Output
            End Enum
            ''' <summary>
            ''' maximum time the network should be executed until a trained network is found (used in training)
            ''' </summary>
            ''' <remarks></remarks>
            Enum TransferFunctionType
                none
                sigmoid
                HyperbolTangent
                BinaryThreshold
                RectifiedLinear
                Logistic
                StochasticBinary
                Gaussian
                Signum
            End Enum
            ''' <summary>
            ''' Each layer consists of nodes (neurons) these are each individual. all layers contain nodes,
            ''' training cases will also use nodes as inputs to the neural network
            ''' </summary>
            ''' <remarks></remarks>
            Public Structure Neuron
                ''' <summary>
                ''' The input of the node is the collective sum of the inputs and their respective weights
                ''' </summary>
                ''' <remarks></remarks>
                Public input As Double
                ''' <summary>
                ''' the output of the node is also relational to the transfer function used
                ''' </summary>
                ''' <remarks></remarks>
                Public output As Double
                ''' <summary>
                ''' There is a value attached with dendrite called weight.
                ''' The weight associated with a dendrites basically determines the importance of incoming value.
                ''' A weight with larger value determines that the value from that particular neuron is of higher significance.
                ''' To achieve this what we do is multiply the incoming value with weight.
                ''' So no matter how high the value is, if the weight is low the multiplication yields the final low value.
                ''' </summary>
                ''' <remarks></remarks>
                Public weight As Double
                ''' <summary>
                ''' the error that is produced respective to the output node, is required for calculations of the new weights
                ''' </summary>
                ''' <remarks></remarks>
                Public NeuronError As Double
            End Structure
            ''' <summary>
            ''' Each layer consists of neurons(nodes) the training cases also use an input layer and an output layer
            ''' </summary>
            ''' <remarks></remarks>
            Public Structure Layer
                ''' <summary>
                ''' Collection of nodes
                ''' </summary>
                ''' <remarks></remarks>
                Public Nodes As List(Of Neuron)
                ''' <summary>
                ''' Activation function used by the nodes in the layer
                ''' </summary>
                ''' <remarks></remarks>
                Public ActivationFunction As TransferFunctionType
                ''' <summary>
                ''' Type of layer (Input, Hidden, Output)
                ''' </summary>
                ''' <remarks></remarks>
                Public LayerType As LayerType
                ''' <summary>
                ''' The number of nodes is stored to make iteration easier
                ''' </summary>
                ''' <remarks></remarks>
                Public NumberOfNodes As Integer
            End Structure
            ''' <summary>
            ''' layer takes the inputs(the values you pass) and forwards it to hidden layer.
            ''' You can just imagine input layer as a group of neurons whose sole task is to pass the numeric inputs to the next level.
            '''  Input layer never processes data, it just hands over it.
            ''' </summary>
            ''' <remarks>there is only one layer for the input</remarks>
            Public InputLayer As Layer
            ''' <summary>
            ''' Middle layer: This layer is the real thing behind the network. Without this layer, network would not be capable of solving complex problems.
            ''' There can be any number or middle or hidden layers. But, for most of the tasks, one is sufficient. The number of neurons in this layer is crucial.
            ''' There is no formula for calculating the number, just hit and trial works.
            ''' This layer takes the input from input layer, does some calculations and forwards to the next layer, in most cases it is the output layer.
            ''' </summary>
            ''' <remarks>in a deep belief network there can be many hidden layers</remarks>
            Public HiddenLayers As List(Of Layer)
            ''' <summary>
            ''' Output layer: This layer consists of neurons which output the result to you. This layer takes the value from the previous layer,
            ''' does calculations and gives the final result. Basically,
            ''' this layer is just like hidden layer but instead of passing values to the next layer, the values are treated as output.
            ''' </summary>
            ''' <remarks>there is only one layer for the output</remarks>
            Public OutputLayer As Layer
            ''' <summary>
            ''' Initial Weights can be determined by the number of hidden nodes and 
            ''' the number of input nodes 
            ''' this is a rule of thumb
            ''' </summary>
            ''' <param name="InputL">Number of Input Nodes</param>
            ''' <param name="InputH">Number of Hidden Nodes</param>
            ''' <returns>a random weight amount which can be used as initial weights</returns>
            ''' <remarks></remarks>
            Private Shared Function CreateRandWeight(ByRef InputL As Integer, ByRef InputH As Integer) As Integer
                Randomize()
                Dim value As Integer = CInt(Int((InputH * Rnd()) + InputL))
                Return value
            End Function
            ''' <summary>
            ''' The number of hidden nodes to become effective is actually unknown yet a simple calculation 
            ''' can be used to determine an initial value which should be effective; 
            ''' </summary>
            ''' <param name="NumbeOfInputNodes">the number of input node used in the network</param>
            ''' <param name="NumberOfOutputNodes">the number of out put nodes in the network</param>
            ''' <returns>a reasonable calculation for hidden nodes</returns>
            ''' <remarks>Deep layer networks have multiple hidden layers with varied number of nodes</remarks>
            Private Shared Function CalculateNumberOfHiddenNodes(ByRef NumbeOfInputNodes As Integer, ByRef NumberOfOutputNodes As Integer) As Integer
                CalculateNumberOfHiddenNodes = NumbeOfInputNodes + NumberOfOutputNodes / 2
                If CalculateNumberOfHiddenNodes < NumberOfOutputNodes Then CalculateNumberOfHiddenNodes = NumberOfOutputNodes
            End Function
            ''' <summary>
            ''' the output for the layer can be provided as an input 
            ''' to each node in the next layer
            ''' </summary>
            ''' <param name="nlayer">Layer to be evaluated</param>
            ''' <returns>total output for the layer</returns>
            ''' <remarks>
            ''' LayerOutput = SumOfNodeOutputs</remarks>
            Public Shared Function SumLayerOutputs(ByRef nlayer As Layer) As Double
                Dim LayerOutput As Double = 0

                For Each node As Neuron In nlayer.Nodes
                    node.output = NodeTotal(node)
                    LayerOutput += node.output
                Next
                'to be passed to next layer
                Return LayerOutput
            End Function
            ''' <summary>
            ''' Produces a node total which can be fed to the activation function
            ''' </summary>
            ''' <param name="Node">Node to be calculated</param>
            ''' <returns>Node input * Node Weight</returns>
            ''' <remarks></remarks>
            Private Shared Function NodeTotal(ByRef Node As Neuron) As Double
                Dim Sum As Double = 0
                Sum = Sum + (Node.input * Node.weight)
                Return Sum
            End Function
            ''' <summary>
            ''' Activates each node in the layer
            ''' </summary>
            ''' <param name="Hlayer">Layer to be activated (Hidden or Output)</param>
            ''' <returns>activated layer</returns>
            ''' <remarks>layer to be summed to be passed to the inputs of the next layer</remarks>
            Public Shared Function ActivateLayer(ByRef Hlayer As Layer) As Layer
                For Each node As Neuron In Hlayer.Nodes
                    node = ActivateNode(node, Hlayer.ActivationFunction)
                Next
                Return Hlayer
            End Function
            ''' <summary>
            ''' Activates Node and sets the output for the node
            ''' </summary>
            ''' <param name="Node">Node to be activated</param>
            ''' <param name="Activation">Activation Function</param>
            ''' <returns>Activated Node</returns>
            ''' <remarks>ActivationFunction(Node.input * Node.weight)</remarks>
            Private Shared Function ActivateNode(ByRef Node As Neuron, ByRef Activation As TransferFunctionType) As Neuron
                Dim Sum As Double = 0
                Sum = NodeTotal(Node)
                Node.output = EvaluateTransferFunct(Activation, Sum)
                Return Node
            End Function
            ''' <summary>
            ''' Produces a sum of the weights of the layer
            ''' </summary>
            ''' <param name="nlayer"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Public Shared Function SumWeights(ByRef nlayer As Layer) As Double
                Dim Sum As Double = 0
                For Each node As Neuron In nlayer.Nodes
                    Sum = Sum + node.weight
                Next
                Return Sum
            End Function
            'Evaluate
            Public Shared Function EvaluateTransferFunct(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                EvaluateTransferFunct = 0
                Select Case TransferFunct
                    Case TransferFunctionType.none
                        Return Input
                    Case TransferFunctionType.sigmoid
                        Return Sigmoid(Input)
                    Case TransferFunctionType.HyperbolTangent
                        Return HyperbolicTangent(Input)
                    Case TransferFunctionType.BinaryThreshold
                        Return BinaryThreshold(Input)
                    Case TransferFunctionType.RectifiedLinear
                        Return RectifiedLinear(Input)
                    Case TransferFunctionType.Logistic
                        Return Logistic(Input)
                    Case TransferFunctionType.Gaussian
                        Return Gaussian(Input)
                    Case TransferFunctionType.Signum
                        Return Signum(Input)
                End Select
            End Function
            Public Shared Function EvaluateTransferFunctionDerivative(ByRef TransferFunct As TransferFunctionType, ByRef Input As Double) As Integer
                EvaluateTransferFunctionDerivative = 0
                Select Case TransferFunct
                    Case TransferFunctionType.none
                        Return Input
                    Case TransferFunctionType.sigmoid
                        Return SigmoidDerivitive(Input)
                    Case TransferFunctionType.HyperbolTangent
                        Return HyperbolicTangentDerivative(Input)
                    Case TransferFunctionType.Logistic
                        Return LogisticDerivative(Input)
                    Case TransferFunctionType.Gaussian
                        Return GaussianDerivative(Input)
                End Select
            End Function
            'Linear Neurons
            ''' <summary>
            ''' in a liner neuron the weight(s) represent unknown values to be determined
            ''' the outputs could represent the known values of a meal and the
            ''' inputs the items in the meal and the
            ''' weights the prices of the individual items
            ''' There are no hidden layers
            ''' </summary>
            ''' <remarks>answers are determined by determining the weights of the linear neurons
            ''' the delta rule is used as the learning rule: Weight = Learning rate * Input * LocalError of neuron</remarks>
            Private Shared Function Linear(ByRef value As Double) As Double
                ' Output = Bias + (Input*Weight)
                Return value
            End Function
            ''' <summary>
            ''' the step function rarely performs well except in some rare cases with (0,1)-encoded binary data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Private Shared Function BinaryThreshold(ByRef Value As Double) As Double

                ' Z = Bias+ (Input*Weight)
                'TransferFunction
                'If Z > 0 then Y = 1
                'If Z < 0 then y = 0

                If Value < 0 = True Then

                    Return 0
                Else
                    Return 1
                End If
            End Function
            Private Shared Function RectifiedLinear(ByRef Value As Double) As Double
                'z = B + (input*Weight)
                'If Z > 0 then output = z
                'If Z < 0 then output = 0
                If Value < 0 = True Then

                    Return 0
                Else
                    Return Value
                End If
            End Function
            Private Shared Function StochasticBinary(ByRef value As Double) As Double
                'Uncreated
                Return value
            End Function
            'Non Linear neurons
            Private Shared Function Logistic(ByRef Value As Double) As Double
                'z = bias + (sum of all inputs ) * (input*weight)
                'output = Sigmoid(z)
                'derivative input = z/weight
                'derivative Weight = z/input
                'Derivative output = output*(1-Output)
                'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                Return 1 / 1 + Math.Exp(-Value)
            End Function
            Private Shared Function LogisticDerivative(ByRef Value As Double) As Double
                'z = bias + (sum of all inputs ) * (input*weight)
                'output = Sigmoid(z)
                'derivative input = z/weight
                'derivative Weight = z/input
                'Derivative output = output*(1-Output)
                'learning rule = Sum of total training error* derivative input * derivative output * rootmeansquare of errors

                Return Logistic(Value) * (1 - Logistic(Value))
            End Function
            Private Shared Function Gaussian(ByRef x As Double) As Double
                Gaussian = Math.Exp((-x * -x) / 2)
            End Function
            Private Shared Function GaussianDerivative(ByRef x As Double) As Double
                GaussianDerivative = Gaussian(x) * (-x / (-x * -x))
            End Function
            Private Shared Function HyperbolicTangent(ByRef Value As Double) As Double
                '    TanH(x) = (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x))

                Return Math.Tanh(Value)
            End Function
            Private Shared Function HyperbolicTangentDerivative(ByRef Value As Double) As Double
                HyperbolicTangentDerivative = 1 - (HyperbolicTangent(Value) * HyperbolicTangent(Value)) * Value
            End Function
            ''' <summary>
            ''' the log-sigmoid function constrains results to the range (0,1),
            ''' the function is sometimes said to be a squashing function in neural network literature.
            ''' It is the non-linear characteristics of the log-sigmoid function (and other similar activation functions)
            ''' that allow neural networks to model complex data.
            ''' </summary>
            ''' <param name="Value"></param>
            ''' <returns></returns>
            ''' <remarks>1 / (1 + Math.Exp(-Value))</remarks>
            Private Shared Function Sigmoid(ByRef Value As Integer) As Double
                'z = Bias + (Input*Weight)
                'Output = 1/1+e**z
                Return 1 / (1 + Math.Exp(-Value))
            End Function
            Private Shared Function SigmoidDerivitive(ByRef Value As Integer) As Double
                Return Sigmoid(Value) * (1 - Sigmoid(Value))
            End Function
            Private Shared Function Signum(ByRef Value As Integer) As Double
                'z = Bias + (Input*Weight)
                'Output = 1/1+e**z
                Return Math.Sign(Value)
            End Function
        End Structure

        ''' <summary>
        ''' Training case to be trained 
        ''' </summary>
        ''' <remarks>expected output is used to check network error</remarks>
        Public Structure NeuralDecision
            Dim Inputs As List(Of Integer)
            Dim ExpectedOutput As List(Of Integer)
            Dim LearnedOutput As List(Of Integer)
        End Structure
        'Parameters of the network
        ''' <summary>
        ''' Sets the parameters required by the network to be created
        ''' </summary>
        ''' <param name="NumberOfInputs"></param>
        ''' <param name="NumberOfOutputs"></param>
        ''' <param name="OutputLayerFunctionType"></param>
        ''' <param name="HiddenLayerFunctionType"></param>
        ''' <returns></returns>
        ''' <remarks></remarks>
        Private Function SetParameters(ByRef NumberOfInputs As Integer, ByRef NumberOfOutputs As Integer,
                                      ByRef OutputLayerFunctionType As NeuralNetwork.TransferFunctionType,
                                      ByRef HiddenLayerFunctionType As NeuralNetwork.TransferFunctionType)

            Dim Params As New NeuralNetwork.NeuralNetworkParameters
            Params.NumberOfInputs = NumberOfInputs
            Params.NumberOfOutputs = NumberOfOutputs
            Params.HiddenLayerFunctionType = HiddenLayerFunctionType
            Params.OutputLayerFunctionType = OutputLayerFunctionType
            Return Params
        End Function
        Private mCreatedNetwork As New NeuralNetwork
        Public ReadOnly Property CreatedNetwork As NeuralNetwork
            Get
                Return mCreatedNetwork
            End Get
        End Property
        '1. Create(returns created NET)
        ''' <summary>
        ''' Creates a neural network
        ''' </summary>
        ''' <param name="inputs">number of input nodes</param>
        ''' <param name="outputs">number of output nodes</param>
        ''' <remarks>returns a created network</remarks>
        Public Sub New(ByRef inputs As Integer, ByRef outputs As Integer)
            '1 SetParameters
            Dim NetworkParams = SetParameters(inputs, outputs,
                                              NeuralNetwork.TransferFunctionType.RectifiedLinear, NeuralNetwork.TransferFunctionType.sigmoid)
            '2. Create network
            Dim CreateNN As New ClassCreateNetwork(NetworkParams)
            mCreatedNetwork = CreateNN.NeuralNet
        End Sub
        'Train(returns trained network)
        Public Function TrainNetwork(ByRef NewCase As NeuralDecision, ByRef Epochs As Integer,
                                ByRef Threshold As Integer, ByRef LearningRate As Integer, NewNet As NeuralNetwork) As NeuralNetwork

            Dim Training As New ClassTrainNetwork(NewNet, Epochs, Threshold, NewCase.Inputs, NewCase.ExpectedOutput, LearningRate)
            Return Training.NeuralNet
        End Function
        'Execute(returns answer to input case)
        Public Function ExecuteNetwork(ByRef NewCase As NeuralDecision, ByRef TrainedNetwork As NeuralNetwork) As NeuralDecision
            Dim Executing As New ClassTrainNetwork(TrainedNetwork, NewCase.Inputs)
            NewCase.LearnedOutput = Executing.NetworkOutput
            Return NewCase
        End Function
        'Classes required by the Class
        ''' <summary>
        ''' Creates the neural network
        ''' </summary>
        ''' <remarks></remarks>
        ''' 
        Private Class ClassCreateNetwork
            Private mNeuralNet As New NeuralNetwork
            Public ReadOnly Property NeuralNet As NeuralNetwork
                Get
                    Return mNeuralNet
                End Get
            End Property
            Public Sub New(ByRef NetworkParameters As NeuralNetwork.NeuralNetworkParameters)
                mNeuralNet = NeuralNetwork.CreateNeuralNetwork(NetworkParameters)
            End Sub
        End Class

        ''' <summary>
        ''' When training the network the network needs to be forward propagated 
        ''' then the errors need to be reduced by back propagation 
        ''' then re- forward propagated (epoch)
        ''' </summary>
        ''' <remarks></remarks>
        Private Class ClassTrainNetwork
            Private Threshold As Integer
            Private LearningRate As Integer
            Private Epochs As Integer
            Private mNerualNet As NeuralNetwork
            Private mOutputs As List(Of Integer)
            ''' <summary>
            ''' returns the output of the network
            ''' </summary>
            ''' <value></value>
            ''' <returns></returns>
            ''' <remarks>output is populated when the network has been executed</remarks>
            Public ReadOnly Property NetworkOutput As List(Of Integer)
                Get
                    Return mOutputs
                End Get
            End Property
            ''' <summary>
            ''' Returns the trained network
            ''' </summary>
            ''' <value></value>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Public ReadOnly Property NeuralNet As NeuralNetwork
                Get
                    Return mNerualNet
                End Get
            End Property
            ''' <summary>
            ''' the output from the training set is measured against the output from the neural network
            ''' this cost function produces a sum of the squared errors 
            ''' which can be used to find new weights for the neural network
            ''' </summary>
            ''' <param name="nOutput">Output from Neural Network</param>
            ''' <param name="ExpectedOutput">Expected Output from training set</param>
            ''' <returns>The Sum of the squared errors * 0.5</returns>
            ''' <remarks>Cost function for gradient descent</remarks>
            Private Function CheckError(ByRef nOutput As List(Of Integer),
                                  ByRef ExpectedOutput As List(Of Integer)) As Integer
                Dim count As Integer = 0
                Dim cost As Integer = 0
                Dim SquErr As New Integer
                Dim SumSquaredErr As New Integer

                For Each nOut As Integer In nOutput
                    cost = CheckNodeErr(nOut, ExpectedOutput(count))

                    SquErr = cost * cost
                    SumSquaredErr += SquErr
                    count += 1
                Next
                Return SumSquaredErr
            End Function
            ''' <summary>
            ''' Returns the error
            ''' </summary>
            ''' <param name="Recieved"></param>
            ''' <param name="Expected"></param>
            ''' <returns></returns>
            ''' <remarks></remarks>
            Private Function CheckNodeErr(ByRef Recieved As Double, ByRef Expected As Double) As Double
                Return Expected - Recieved
            End Function
            ''' <summary>
            ''' used to train the network
            ''' </summary>
            ''' <param name="NNet">neural network</param>
            ''' <param name="mEpochs">amount of times to execute the network</param>
            ''' <param name="mthreshold">error threshold</param>
            ''' <param name="inputs">Training case input</param>
            ''' <param name="ExpectedOutput">Expected output of case</param>
            ''' <param name="mLearningRate">rate of learning</param>
            ''' <remarks>each training case is to be trained , the more cases the better the network performance
            ''' the trained network is returned</remarks>
            Public Sub New(ByRef NNet As NeuralNetwork, ByRef mEpochs As Integer, ByRef mthreshold As Integer,
                           ByRef inputs As List(Of Integer), ByRef ExpectedOutput As List(Of Integer), ByRef mLearningRate As Integer)
                Threshold = mthreshold
                LearningRate = mLearningRate
                mNerualNet = Train(NNet, inputs, ExpectedOutput)
            End Sub
            ''' <summary>
            ''' Used to execute the trained network
            ''' </summary>
            ''' <remarks>the output from the executed network is returned</remarks>
            Public Sub New(ByRef Nnet As NeuralNetwork, ByRef inputs As List(Of Integer))
                Dim forward As New ForwardPropagation(inputs, Nnet)
                mOutputs = forward.Outputs
            End Sub
            Private Function Train(ByRef NNet As NeuralNetwork, ByRef inputs As List(Of Integer),
                                  ByRef ExpectedOutput As List(Of Integer)) As NeuralNetwork
                For i = 1 To Epochs
                    Dim Forward As New ForwardPropagation(inputs, NNet)
                    Dim NetworkOutput = Forward.Outputs
                    Dim NetworkError = CheckError(NetworkOutput, ExpectedOutput)
                    If NetworkError > Threshold = True Then
                        Dim Backward As New BackwardPropagation(NNet, LearningRate, NetworkError)
                    End If
                Next
                Return NNet
            End Function
            ''' <summary>
            ''' Called to execute the network
            ''' </summary>
            ''' <remarks></remarks>
            Private Class ForwardPropagation
                Private mOutputs As List(Of Integer)
                Public ReadOnly Property Outputs As List(Of Integer)
                    Get
                        Return mOutputs
                    End Get
                End Property
                Public Sub New(ByRef Inputs As List(Of Integer), ByRef NeuralNet As NeuralNetwork)
                    mOutputs = ForwardProp(Inputs, NeuralNet)
                End Sub
                ''' <summary>
                ''' forward propagation of network
                ''' </summary>
                ''' <param name="nInput">Input to neural network</param>
                ''' <param name="nn">Neural network</param>
                ''' <returns>Output generated by network</returns>
                ''' <remarks>Output of network may not be correct until trained</remarks>
                Private Function ForwardProp(ByRef nInput As List(Of Integer), ByRef nn As NeuralNetwork) As List(Of Integer)
                    'Forward Propagation: 
                    'the initial execution of the network produces a result which may not be correct 
                    'This means that training the network still needs to be accomplished

                    '1. Get input(array) <Input>
                    '2. node total(output)
                    '3. sum layer pass to input hidden
                    '4. node total
                    '5. activate hidden
                    '6. Pass Layer
                    '7. repeat 2. (if required) deep learning
                    '8. sum layer pass to input output
                    '9. activate output
                    '10. Get Output(array)<Output>

                    Dim nOutput As New List(Of Integer)
                    Dim count As Integer = 0

                    'L1. Input Layer
                    Dim LayerOutputs As New Integer
                    '1. Get Inputs

                    For Each node As NeuralNetwork.Neuron In nn.InputLayer.Nodes
                        node.input = nInput(count)
                        count += 1
                    Next
                    '2. GetLayerOutputs
                    LayerOutputs = NeuralNetwork.SumLayerOutputs(nn.InputLayer)

                    'L2. Hidden Layer(s)
                    '3. PassLayerOutputs 
                    For Each hlayer As NeuralNetwork.Layer In nn.HiddenLayers
                        For Each node As NeuralNetwork.Neuron In hlayer.Nodes
                            node.input = LayerOutputs
                        Next
                        '4. activate layer
                        hlayer = NeuralNetwork.ActivateLayer(hlayer)
                        '5. GetLayerOutputs
                        LayerOutputs = NeuralNetwork.SumLayerOutputs(hlayer)
                    Next

                    'L3. Output Layer
                    'PassLayerOutputs
                    For Each Node As NeuralNetwork.Neuron In nn.OutputLayer.Nodes
                        Node.input = LayerOutputs
                    Next
                    'activate layer
                    nn.OutputLayer = NeuralNetwork.ActivateLayer(nn.OutputLayer)
                    count = 0

                    'Get Output 
                    For Each node As NeuralNetwork.Neuron In nn.OutputLayer.Nodes
                        nOutput(count) = node.output
                    Next

                    Return nOutput
                End Function
            End Class
            ''' <summary>
            ''' Called to recalculate the network Weights
            ''' </summary>
            ''' <remarks></remarks>
            Private Class BackwardPropagation
                Private mNerualNet As NeuralNetwork
                Public ReadOnly Property NeuralNet As NeuralNetwork
                    Get
                        Return mNerualNet
                    End Get
                End Property
                Public Sub New(ByRef NNet As NeuralNetwork,
                               ByRef learningrate As Integer,
                               ByRef NetworkOutputError As Integer)
                    mNerualNet = BackProp(NNet, learningrate, NetworkOutputError)
                End Sub
                ''' <summary>
                ''' Back Propagation Changes the weights 
                ''' </summary>
                ''' <param name="NN">Current neural network</param>
                ''' <param name="learningrate">rate of change to weights</param>
                ''' <param name="NeuronError">error generated by forward propagation</param>
                ''' <remarks></remarks>
                Private Function BackProp(ByRef NN As NeuralNetwork, ByRef learningrate As Integer,
                                          ByRef NeuronError As Integer) As NeuralNetwork
                    Dim DeltaOutput As Double = 0.0
                    Dim DeltaInput As Double = 0.0
                    Dim DeltaHidden As Double = 0.0
                    Dim SumOfDeltaOutput As Integer = 0
                    Dim SumOfDeltaHidden As Integer = 0
                    Dim SumOfHiddenLayerWeights As Integer = 0

                    'Starts with the Output layer going backwards
                    For Each Node As NeuralNetwork.Neuron In NN.OutputLayer.Nodes
                        DeltaOutput = NeuronError * NeuralNetwork.EvaluateTransferFunctionDerivative(Node.input, NN.OutputLayer.ActivationFunction)
                        'NewWeight=weight+learning-Rate*node.output*DeltaOutput
                        Node.weight = Node.weight + learningrate * Node.output * DeltaOutput
                        SumOfDeltaOutput += DeltaOutput
                    Next


                    'then the hidden layer
                    For Each hlayer As NeuralNetwork.Layer In NN.HiddenLayers
                        For Each Node As NeuralNetwork.Neuron In hlayer.Nodes
                            'deltaHidden = SigmoidDerivitive(Node.input) * SumOfDeltaOutput*weights
                            DeltaHidden = NeuralNetwork.EvaluateTransferFunctionDerivative(Node.input, hlayer.ActivationFunction) * NeuralNetwork.SumWeights(NN.OutputLayer) * SumOfDeltaOutput
                            Node.weight = Node.weight + learningrate * Node.output * DeltaHidden
                            SumOfDeltaHidden = SumOfDeltaHidden + DeltaHidden
                            SumOfHiddenLayerWeights += Node.weight
                        Next
                    Next

                    'Then the input layer
                    For Each node As NeuralNetwork.Neuron In NN.InputLayer.Nodes
                        DeltaInput = SumOfHiddenLayerWeights * SumOfDeltaHidden
                        node.weight = node.weight + learningrate * node.output * DeltaInput
                    Next


                    Return NN
                End Function
            End Class
        End Class
    End Class
