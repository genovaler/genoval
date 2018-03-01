pragma solidity ^0.4.16;
contract GenovalToken {

    mapping(address => uint256) balances;
    uint256 totalSupply_;
    address origin_;
    uint256 maxSupply_ = 1000;

    mapping(address => mapping(address => uint256)) allowed;

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address _spender, uint256 _value);
    event Mint(address indexed _to, uint256 _amount);
    
    function GenovalToken() public {
        //address a = address(0xca35b7d915458ef540ade6068dfe2f44e8fa733c);
        origin_ = tx.origin;
        balances[origin_] = 100;
        totalSupply_ = 100;
    }
    
    function mint(address _to, uint256 amount) public returns (bool success) {
        totalSupply_ = add(totalSupply_, amount);
        require(totalSupply_ <= maxSupply_);
        balances[_to] = add(balances[_to], amount);
        Mint(_to, amount);
        return true;
    }
    

    function totalSupply() public returns (uint256 t) {
        return totalSupply_;
    }

    function balanceOf (address _owner) public constant returns (uint256 balance) {
        return balances[_owner];
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(_to != address(0));
        require(_value <= balances[msg.sender]);

        assert(balances[msg.sender] >= _value);
        balances[msg.sender] -= _value;
        uint256 c = balances[_to] + _value;
        assert(c > balances[_to]);
        balances[_to] += _value;
        Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
        require(_to != address(0));
        require(_value <= balances[_from]);
        require(_value <= allowed[_from][msg.sender]);

        balances[_from] = sub(balances[_from], _value);
        balances[_to] = add(balances[_to], _value);
        allowed[_from][msg.sender] = sub(allowed[_from][msg.sender], _value);

        Transfer(_from, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowed[msg.sender][_spender] = _value;
        Approval(msg.sender, _spender, _value);
        return true;
    }

    function allowance(address _owner, address _spender) public constant returns (uint256 remaining) {
        return allowed[_owner][_spender];
    }

    function increaseApproval(address _spender, uint256 _addedValue) public returns(bool) {
        allowed[msg.sender][_spender] = allowed[msg.sender][_spender] + _addedValue;
        Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }

    function decreaseApproval(address _spender, uint256 _substractValue)public returns(bool) {
        uint256 oldValue = allowed[msg.sender][_spender];

        if (oldValue < _substractValue) {
            allowed[msg.sender][_spender] = 0;
        } else {
            allowed[msg.sender][_spender] = sub(allowed[msg.sender][_spender], _substractValue);
        }

        Approval(msg.sender, _spender, allowed[msg.sender][_spender]);
        return true;
    }

    function sub(uint256 a, uint256 b) private constant returns (uint256) {
        assert(a >= b);
        return a - b;
    }

    function add(uint256 a, uint256 b) private constant returns (uint256) {
        uint256 d = a + b;
        assert(d > a);
        return d;
    }
}
